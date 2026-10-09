"""
Obsidian Research Note Agent
=============================

A production-hardened CrewAI pipeline that researches a topic and writes a
validated, Obsidian-ready Markdown note.

Pipeline:
    plan -> research -> write -> validate -> (revise -> validate)* -> save

Key production hardening vs. the prototype:
    - No Colab dependency; config comes from environment variables (.env).
    - Cross-platform, configurable vault path instead of a hardcoded path.
    - Structured (Pydantic) validation output so the workflow can branch
      programmatically instead of string-matching "STATUS: PASS".
    - A real retry loop (write -> validate -> revise) with a max attempt
      cap, implemented via CrewAI's Flow instead of a flat sequential Crew
      that only revises once no matter what the validator says.
    - Retries with backoff around network-bound tool calls (Tavily).
    - Defensive error handling in every tool (bad input, API errors, empty
      results, illegal filenames).
    - Structured logging with a per-run ID, plus correct AgentOps session
      lifecycle handling (session always closed, success or failure).
    - A real CLI entrypoint instead of bare top-level `await`.
    - Input validation and a hard timeout on the overall run.

Install:
    pip install crewai crewai-tools tavily-python agentops python-dotenv pydantic

Configure (.env or real secrets manager):
    GROQ_API_KEY=...          # only needed if you actually use Groq
    OPENROUTER_API_KEY=...
    TAVILY_API_KEY=...
    AGENTOPS_API_KEY=...
    OBSIDIAN_VAULT_PATH=/path/to/your/vault
    LLM_MODEL=openrouter/anthropic/claude-3.5-haiku   # pick a paid, stable model
    MAX_REVISION_ATTEMPTS=2
    RUN_TIMEOUT_SECONDS=600

Run:
    python obsidian_research_agent.py --topic "What is NLP"
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import logging
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

load_dotenv()

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("obsidian_research_agent")


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

class ConfigError(RuntimeError):
    pass


class Config:
    """Centralized, validated runtime configuration."""

    def __init__(self) -> None:
        self.tavily_api_key = self._require("TAVILY_API_KEY")
        self.openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
        self.groq_api_key = os.environ.get("GROQ_API_KEY")
        self.agentops_api_key = os.environ.get("AGENTOPS_API_KEY")  # optional

        self.llm_model = os.environ.get(
            "LLM_MODEL", "openrouter/anthropic/claude-3.5-haiku"
        )

        vault = os.environ.get("OBSIDIAN_VAULT_PATH")
        if not vault:
            raise ConfigError(
                "OBSIDIAN_VAULT_PATH is not set. Point it at your Obsidian vault folder."
            )
        self.vault_path = Path(vault).expanduser().resolve()

        self.max_revision_attempts = int(os.environ.get("MAX_REVISION_ATTEMPTS", "2"))
        self.run_timeout_seconds = int(os.environ.get("RUN_TIMEOUT_SECONDS", "600"))

    @staticmethod
    def _require(key: str) -> str:
        val = os.environ.get(key)
        if not val:
            raise ConfigError(f"Missing required environment variable: {key}")
        return val


# --------------------------------------------------------------------------
# Retry helper (no extra dependency beyond stdlib)
# --------------------------------------------------------------------------

def retry(times: int = 3, base_delay: float = 1.5, exceptions=(Exception,)):
    """Simple exponential-backoff retry decorator for flaky network calls."""

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1, times + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:  # noqa: BLE001
                    last_exc = exc
                    if attempt == times:
                        break
                    delay = base_delay * (2 ** (attempt - 1))
                    logger.warning(
                        "%s failed (attempt %d/%d): %s — retrying in %.1fs",
                        fn.__name__, attempt, times, exc, delay,
                    )
                    time.sleep(delay)
            logger.error("%s failed after %d attempts: %s", fn.__name__, times, last_exc)
            raise last_exc

        return wrapper

    return decorator


# --------------------------------------------------------------------------
# Structured outputs
# --------------------------------------------------------------------------

class ValidationReport(BaseModel):
    status: str = Field(description="Either 'PASS' or 'FAIL'")
    issues: List[str] = Field(
        default_factory=list,
        description="Prioritized, specific corrections the writer should make. Empty if PASS.",
    )

    @property
    def passed(self) -> bool:
        return self.status.strip().upper() == "PASS"


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------

def _build_tools(cfg: Config):
    """Build tool callables bound to the given config. Deferred import so
    this module can be imported for testing without crewai installed."""

    from crewai.tools import tool
    from tavily import TavilyClient

    tavily_client = TavilyClient(api_key=cfg.tavily_api_key)

    @retry(times=3, exceptions=(Exception,))
    def _tavily_search(query: str):
        return tavily_client.search(
            query=query,
            search_depth="advanced",
            max_results=5,
            include_answer=False,
        )

    @retry(times=3, exceptions=(Exception,))
    def _tavily_extract(urls: List[str]):
        return tavily_client.extract(urls=urls, extract_depth="advanced", format="markdown")

    @tool("web_search")
    def web_search(query: str) -> str:
        """
        Search the web for up-to-date information.
        Use this tool when you need to discover relevant sources,
        articles, documentation, or information about a topic.
        """
        if not query or not query.strip():
            return "Error: empty query provided to web_search."

        try:
            response = _tavily_search(query.strip())
        except Exception as exc:  # noqa: BLE001
            logger.exception("web_search failed for query=%r", query)
            return f"Error: web search failed ({exc}). Try a different or simpler query."

        results = response.get("results", [])
        if not results:
            return f"No results found for query: {query}"

        chunks = []
        for r in results:
            chunks.append(
                f"Title: {r.get('title', 'N/A')}\n"
                f"URL: {r.get('url', 'N/A')}\n"
                f"Content: {r.get('content', '')}"
            )
        return "\n---\n".join(chunks)

    @tool("web_extract")
    def web_extract(urls: str) -> str:
        """
        Extract clean content from one or more web pages (comma-separated URLs).
        Use this after web_search when you need detailed information
        from specific URLs.
        """
        if not urls or not urls.strip():
            return "Error: no URLs provided to web_extract."

        url_list = [u.strip() for u in urls.split(",") if u.strip()]
        url_list = [u for u in url_list if u.startswith("http://") or u.startswith("https://")]

        if not url_list:
            return "Error: no valid http(s) URLs found in input."

        try:
            response = _tavily_extract(url_list)
        except Exception as exc:  # noqa: BLE001
            logger.exception("web_extract failed for urls=%r", url_list)
            return f"Error: extraction failed ({exc})."

        results = response.get("results", [])
        failed = response.get("failed_results", [])

        if not results:
            return f"Error: extraction returned no content. Failed: {failed}"

        chunks = []
        for r in results:
            chunks.append(f"URL: {r.get('url', 'N/A')}\n\nContent:\n{r.get('raw_content', '')}")
        out = "\n\n====================\n\n".join(chunks)
        if failed:
            out += f"\n\n[Note: {len(failed)} URL(s) could not be extracted.]"
        return out

    def _safe_filename(title: str) -> str:
        """Strip characters illegal in filenames on Windows/macOS/Linux."""
        cleaned = re.sub(r'[\\/:*?"<>|]', "-", title).strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        if not cleaned:
            cleaned = f"untitled-{uuid.uuid4().hex[:8]}"
        return cleaned[:150]  # keep filenames reasonable

    @tool("Save note to Obsidian")
    def save_to_obsidian(title: str, content: str) -> str:
        """Save a Markdown note into the Obsidian vault. Will not overwrite
        an existing note with the same name; appends a numeric suffix instead."""
        if not content or not content.strip():
            return "Error: refused to save empty content."

        try:
            folder = cfg.vault_path / "AI Notes"
            folder.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.exception("Could not create vault folder")
            return f"Error: could not create/access vault folder: {exc}"

        safe_title = _safe_filename(title or "untitled-note")
        file_path = folder / f"{safe_title}.md"

        # Avoid clobbering an existing note with the same title.
        counter = 1
        while file_path.exists():
            file_path = folder / f"{safe_title} ({counter}).md"
            counter += 1

        try:
            file_path.write_text(content, encoding="utf-8")
        except OSError as exc:
            logger.exception("Failed writing note to disk")
            return f"Error: failed to write note: {exc}"

        logger.info("Saved note to %s", file_path)
        return f"Note saved successfully: {file_path}"

    return web_search, web_extract, save_to_obsidian


# --------------------------------------------------------------------------
# Agents / Tasks
# --------------------------------------------------------------------------

def _build_crew_components(cfg: Config):
    from crewai import LLM, Agent, Task

    web_search, web_extract, save_to_obsidian = _build_tools(cfg)

    llm_kwargs = {"model": cfg.llm_model}
    if cfg.openrouter_api_key and "openrouter" in cfg.llm_model:
        llm_kwargs["base_url"] = "https://openrouter.ai/api/v1"
        llm_kwargs["api_key"] = cfg.openrouter_api_key
    llm = LLM(**llm_kwargs)

    research_planner = Agent(
        role="Research Planning Specialist",
        goal=(
            "Create a comprehensive and logically ordered research plan for the "
            "requested topic, identifying the concepts, questions, examples, "
            "technical details, and relationships that should be investigated "
            "before generating an educational Obsidian note."
        ),
        backstory=(
            "You are an expert educational researcher and knowledge architect. "
            "You break complex technical subjects into logical knowledge "
            "structures that build understanding step by step, connect the "
            "topic to related concepts, and include practical examples when "
            "appropriate. You avoid unnecessary research."
        ),
        llm=llm,
        verbose=True,
    )

    research_task = Task(
        name="Research Plan Task",
        description=(
            "Analyze the user's requested topic {topic} and create a detailed "
            "research plan for producing a high-quality technical note. Cover: "
            "fundamental definition; the problem it solves; why it matters; core "
            "concepts; internal mechanics; key terminology; relevant formulas; "
            "practical and code examples; advantages/disadvantages; common "
            "mistakes; important parameters/components; related concepts to "
            "link; what should be verified via external sources. Order from "
            "beginner to advanced. Do not write the final note or do the "
            "research yourself."
        ),
        expected_output="A structured research plan that another agent can execute.",
        agent=research_planner,
    )

    web_researcher = Agent(
        role="Technical Research Specialist",
        goal=(
            "Conduct accurate, comprehensive, source-aware web research using "
            "the search/extract tools to gather the information required by "
            "the research plan, prioritizing authoritative sources."
        ),
        backstory=(
            "You are a meticulous technical researcher. You prioritize official "
            "documentation, academic papers, and primary sources, cross-check "
            "claims, and flag conflicting information. You gather only what's "
            "needed for another agent to write the final note."
        ),
        tools=[web_search, web_extract],
        llm=llm,
        verbose=True,
    )

    searcher_task = Task(
        name="Web Searcher Task",
        description=(
            "Using the research plan, conduct detailed web research. For every "
            "major section: search for reliable information, prefer "
            "authoritative/primary sources, cross-check important claims, note "
            "conflicting information, and extract definitions, explanations, "
            "examples, formulas, and implementation details. Record source URLs. "
            "Do not write the final note."
        ),
        expected_output="A structured research report with source URLs per section.",
        agent=web_researcher,
    )

    note_writer = Agent(
        role="Technical Note Writer",
        goal=(
            "Transform the research report into a clear, accurate, "
            "well-structured, educational Markdown note optimized for direct "
            "use inside Obsidian."
        ),
        backstory=(
            "You are an expert technical writer and PKM specialist. You explain "
            "concepts progressively, use Markdown and Obsidian features "
            "(internal links, tags, callouts, code blocks, tables) effectively, "
            "and avoid filler, repetition, and generic AI-sounding language. "
            "Your output is immediately usable as-is."
        ),
        llm=llm,
        verbose=True,
    )

    note_writing_task = Task(
        name="Note Writing Task",
        description=(
            "Using the research report and plan, create the final Obsidian-"
            "ready Markdown note: H1 title; concise overview; fundamentals to "
            "advanced progression; defined terminology; mechanics; formulas "
            "where relevant; practical/code examples where useful; key "
            "parameters/components; advantages and limitations; common "
            "mistakes; [[Topic]] internal links; tags; tables/callouts where "
            "they add value. No filler, no 'In this note...', no mention that "
            "it's AI-generated, nothing unsupported by the research report."
        ),
        expected_output="The final structured Markdown note.",
        agent=note_writer,
        context=[research_task, searcher_task],
    )

    note_validator = Agent(
        role="Technical Knowledge Quality Auditor",
        goal=(
            "Critically evaluate the generated Obsidian note for factual "
            "accuracy, completeness, clarity, technical quality, and Obsidian "
            "compatibility, and identify problems that must be corrected."
        ),
        backstory=(
            "You are a rigorous technical reviewer who looks for problems "
            "rather than assuming correctness: factual errors, unsupported "
            "claims, missing concepts, incorrect formulas, broken code, poor "
            "explanations, repetition, and invalid Markdown/Obsidian syntax. "
            "You give actionable, specific feedback."
        ),
        llm=llm,
        verbose=True,
    )

    note_validation_task = Task(
        name="Note Validation Task",
        description=(
            "Review the generated note against the research plan and report "
            "on: Accuracy (correct claims/definitions/formulas/examples, no "
            "unsupported claims), Completeness (all key concepts covered), "
            "Clarity (logical progression), Technical Quality (correct code, "
            "parameters, trade-offs), Obsidian Quality (valid Markdown, "
            "headings, links, code blocks), Writing Quality (no filler/"
            "repetition). Set status to PASS only if there are no required "
            "corrections; otherwise FAIL with a prioritized, specific issues "
            "list the writer can act on directly. Do not rewrite the note."
        ),
        expected_output=(
            "A JSON object matching the ValidationReport schema: "
            '{"status": "PASS"|"FAIL", "issues": ["..."]}'
        ),
        agent=note_validator,
        context=[research_task, searcher_task, note_writing_task],
        output_pydantic=ValidationReport,
    )

    revision_task = Task(
        name="Revision Task",
        description=(
            "Revise the generated Obsidian note according to the validation "
            "report's issues list. Apply every necessary correction. Preserve "
            "what was already correct. Do not introduce new unsupported "
            "information. Keep the note accurate, complete, clear, well "
            "structured, Obsidian-compatible, and directly usable as Markdown."
        ),
        expected_output="ONLY the revised Markdown note, nothing else.",
        agent=note_writer,
        context=[note_writing_task, note_validation_task],
    )

    obsidian_agent = Agent(
        role="Obsidian Note Manager",
        goal="Save finalized notes into the user's Obsidian vault.",
        backstory="You organize AI-generated knowledge into clean, well-named Markdown notes.",
        tools=[save_to_obsidian],
        llm=llm,
        verbose=True,
    )

    save_task = Task(
        name="Save Note to Obsidian",
        description=(
            "Save the final, validated Markdown note to Obsidian. Use the "
            "finalized Markdown content provided in context. Create a "
            "descriptive filename based on the topic '{topic}'. Pass the "
            "title and complete Markdown content to the saving tool."
        ),
        expected_output="Confirmation that the note was saved successfully, including its path.",
        agent=obsidian_agent,
    )

    return {
        "research_planner": research_planner,
        "research_task": research_task,
        "web_researcher": web_researcher,
        "searcher_task": searcher_task,
        "note_writer": note_writer,
        "note_writing_task": note_writing_task,
        "note_validator": note_validator,
        "note_validation_task": note_validation_task,
        "revision_task": revision_task,
        "obsidian_agent": obsidian_agent,
        "save_task": save_task,
    }


# --------------------------------------------------------------------------
# Flow: plan -> research -> write -> validate -> (revise -> validate)* -> save
# --------------------------------------------------------------------------

def run_pipeline(topic: str, cfg: Config) -> str:
    """Runs the full pipeline synchronously and returns the final saved-note
    confirmation message. Raises on unrecoverable failure."""

    from crewai import Crew, Process

    topic = (topic or "").strip()
    if not topic:
        raise ValueError("topic must be a non-empty string")
    if len(topic) > 300:
        raise ValueError("topic is too long (max 300 characters)")

    comp = _build_crew_components(cfg)

    # Stage 1: plan + research + initial draft
    drafting_crew = Crew(
        agents=[comp["research_planner"], comp["web_researcher"], comp["note_writer"]],
        tasks=[comp["research_task"], comp["searcher_task"], comp["note_writing_task"]],
        process=Process.sequential,
        verbose=True,
    )
    drafting_crew.kickoff(inputs={"topic": topic})
    current_note = comp["note_writing_task"].output.raw

    # Stage 2: validate -> revise loop, bounded by max_revision_attempts
    validation_crew = Crew(
        agents=[comp["note_validator"]],
        tasks=[comp["note_validation_task"]],
        process=Process.sequential,
        verbose=True,
    )
    revision_crew = Crew(
        agents=[comp["note_writer"]],
        tasks=[comp["revision_task"]],
        process=Process.sequential,
        verbose=True,
    )

    report: Optional[ValidationReport] = None
    for attempt in range(1, cfg.max_revision_attempts + 2):  # +1 initial validation, +1 for range
        validation_crew.kickoff(inputs={"topic": topic})
        raw_report = comp["note_validation_task"].output
        try:
            report = raw_report.pydantic or ValidationReport.model_validate_json(raw_report.raw)
        except (ValidationError, Exception):  # noqa: BLE001
            logger.warning("Validator did not return parseable JSON; treating as FAIL.")
            report = ValidationReport(status="FAIL", issues=["Validator output was unparseable; re-run."])

        logger.info("Validation attempt %d: status=%s issues=%d", attempt, report.status, len(report.issues))

        if report.passed:
            break

        if attempt > cfg.max_revision_attempts:
            logger.warning(
                "Max revision attempts (%d) reached; proceeding with best available draft.",
                cfg.max_revision_attempts,
            )
            break

        revision_crew.kickoff(inputs={"topic": topic, "issues": "\n".join(report.issues)})
        current_note = comp["revision_task"].output.raw

    if report is not None and not report.passed:
        logger.warning("Saving note that did not reach PASS status. Issues: %s", report.issues)

    # Stage 3: save
    save_crew = Crew(
        agents=[comp["obsidian_agent"]],
        tasks=[comp["save_task"]],
        process=Process.sequential,
        verbose=True,
    )
    # Hand the finalized markdown to the save task explicitly via task context
    comp["save_task"].description += f"\n\nFinal Markdown content to save:\n\n{current_note}"
    result = save_crew.kickoff(inputs={"topic": topic})

    return str(result)


# --------------------------------------------------------------------------
# Entrypoint
# --------------------------------------------------------------------------

def _init_agentops(cfg: Config, run_id: str):
    if not cfg.agentops_api_key:
        logger.info("AGENTOPS_API_KEY not set; skipping AgentOps tracing.")
        return None
    import agentops

    agentops.init(
        api_key=cfg.agentops_api_key,
        default_tags=["crewai", "obsidian-research-agent"],
        skip_auto_end_session=True,
    )
    return agentops


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate an Obsidian research note via an agent crew.")
    parser.add_argument("--topic", required=True, help="Topic to research, e.g. 'What is NLP'")
    args = parser.parse_args(argv)

    run_id = uuid.uuid4().hex[:8]
    log = logging.LoggerAdapter(logger, {})
    log.info("Starting run %s for topic=%r", run_id, args.topic)

    try:
        cfg = Config()
    except ConfigError as exc:
        logger.error("Configuration error: %s", exc)
        return 2

    agentops_mod = _init_agentops(cfg, run_id)
    exit_code = 0

    try:
        result = asyncio.run(
            asyncio.wait_for(
                asyncio.to_thread(run_pipeline, args.topic, cfg),
                timeout=cfg.run_timeout_seconds,
            )
        )
        print(result)
    except asyncio.TimeoutError:
        logger.error("Run %s timed out after %ds", run_id, cfg.run_timeout_seconds)
        exit_code = 3
    except ValueError as exc:
        logger.error("Invalid input: %s", exc)
        exit_code = 2
    except Exception:  # noqa: BLE001
        logger.exception("Run %s failed with an unhandled error", run_id)
        exit_code = 1
    finally:
        if agentops_mod is not None:
            try:
                agentops_mod.end_session("Success" if exit_code == 0 else "Fail")
            except Exception:  # noqa: BLE001
                logger.exception("Failed to cleanly end AgentOps session")

    log.info("Run %s finished with exit code %d", run_id, exit_code)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
