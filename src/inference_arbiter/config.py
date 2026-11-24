"""Config loading and validation for the Phase 1 router.

models.yaml is the only place models, tiers and thresholds are defined.
Secrets never live in the config: providers name an environment variable.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

DEFAULT_CONFIG_PATH = Path("config/models.yaml")
DEFAULT_ENV_PATH = Path(".env")

TRIAGE_TIER = "triage"
REQUIRED_TIERS = ("triage", "local", "code", "local_heavy", "mid", "frontier")
# Tiers the triage classifier is allowed to pick. local_heavy is never auto-routed.
ROUTABLE_TIERS = ("local", "code", "mid", "frontier")

Effort = Literal["low", "medium", "high"]


class ConfigError(Exception):
    """Raised when models.yaml is missing, unreadable or invalid."""


class ProviderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # openrouter_decisions is OpenRouter's Decisions API (typed questions, e.g. TypeSafe Jev).
    # It returns typed answers, not text, so it can only serve a triage or verifier tier.
    kind: Literal["ollama_native", "openrouter", "openrouter_decisions"]
    base_url: str
    api_key_env: str | None = None

    @model_validator(mode="after")
    def _openrouter_needs_key_name(self) -> "ProviderConfig":
        if self.kind in ("openrouter", "openrouter_decisions") and not self.api_key_env:
            raise ValueError(f"an {self.kind} provider must set api_key_env")
        return self


class TierConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    provider: str
    # triage tiers classify the question; answer tiers answer it. The tier named
    # "triage" is always a triage tier; extra triage candidates set role: triage.
    # verifier tiers check a draft (chain mode) or its grounding (openbook mode) and are
    # never answer tiers. An embedding tier turns text into vectors for retrieval.
    role: Literal["answer", "triage", "verifier", "embedding"] = "answer"
    cloud: bool = False
    timeout_s: float = Field(gt=0)
    effort: Effort | None = None
    think: bool | Literal["effort"] | None = None
    num_ctx: int | None = Field(default=None, gt=0)
    # The model's context window in tokens, when known. Stuffed mode refuses a prompt
    # estimated above it instead of sending text the model would silently truncate.
    context_window_tokens: int | None = Field(default=None, gt=0)
    keep_alive: str | None = None
    temperature: float | None = Field(default=None, ge=0)
    max_output_tokens: int | None = Field(default=None, gt=0)
    on_error: str | None = None
    input_price_per_mtok: float | None = Field(default=None, ge=0)
    output_price_per_mtok: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _think_effort_needs_effort(self) -> "TierConfig":
        if self.think == "effort" and self.effort is None:
            raise ValueError("think: effort requires the tier to set effort")
        return self


class ChainConfig(BaseModel):
    """Chain mode: draft, schema gate, verify, at most one repair."""

    model_config = ConfigDict(extra="forbid")

    verifier_tier: str
    repair_tier: str = "frontier"
    # What to do when the verifier call fails, times out or returns an invalid verdict.
    on_verifier_error: Literal["return_unverified", "repair"] = "return_unverified"
    max_repairs: int = Field(default=1, ge=0, le=1)
    # How the repair tier is called when a draft is not supported:
    #   escalate  send only the original question, as a plain answer call (default).
    #             Measured 2026-11-21: carrying the draft, its working and the critique
    #             made each repair cost about 2.5x a direct call.
    #   repair    send the question, the rejected draft and the verifier's critique.
    repair_style: Literal["escalate", "repair"] = "escalate"
    # A model must not verify its own draft. When the drafting tier runs the same model
    # as verifier_tier, this verifier is used instead. Unset: such drafts are reported
    # as unverified (verification_status "unavailable") without calling the verifier.
    alt_verifier_tier: str | None = None


class OpenbookConfig(BaseModel):
    """Openbook mode: load, retrieve, draft with citations, quote check, ground, escalate once.

    Also read by stuffed mode for the source-loading limits.
    """

    model_config = ConfigDict(extra="forbid")

    drafter_tier: str = "mid"
    escalation_tier: str = "frontier"
    grounding_verifier_tier: str = "ground_jev"
    # A model must not judge its own draft. Used when the drafting tier runs the same
    # model as grounding_verifier_tier. Unset: such drafts are returned unverified.
    alt_grounding_verifier_tier: str | None = None
    # What to do when the grounding verifier fails, times out or returns invalid output.
    on_verifier_error: Literal["return_unverified", "escalate"] = "return_unverified"
    # hybrid: fuse the embedding ranking and the lexical (BM25) ranking. embedding or
    # lexical: that ranking alone. Without a working embedding tier all three are lexical.
    retrieval_ranking: Literal["hybrid", "embedding", "lexical"] = "hybrid"
    # Embedding tier used to rank passages. Unset: lexical ranking only.
    embedding_tier: str | None = None
    # Task prefixes the embedding model expects (nomic-embed-text uses these two).
    embed_query_prefix: str = "search_query: "
    embed_document_prefix: str = "search_document: "
    top_k: int = Field(default=6, gt=0)
    # When set, retrieval takes the best-ranked passages until this many estimated tokens
    # are filled, and top_k is ignored. A fixed top_k of 6 lost questions that need one
    # passage from each of two documents (open-book eval case OB-21, 2026-11-22).
    retrieval_token_budget: int | None = Field(default=None, gt=0)
    # The escalation tier gets the passages the first draft cited plus the best-ranked ones,
    # up to this many estimated tokens. None or 0: every passage the drafter saw.
    escalation_token_budget: int | None = Field(default=3000, ge=0)
    # A corpus estimated at or under this many tokens skips retrieval: every passage is used.
    full_context_token_budget: int = Field(default=6000, ge=0)
    max_passage_chars: int = Field(default=1200, gt=0)
    fetch_timeout_s: float = Field(default=20, gt=0)
    max_source_bytes: int = Field(default=2_000_000, gt=0)
    # Fetched pages and embedding vectors are cached here.
    cache_dir: str = "data/cache"
    # A cached page older than this is fetched again.
    page_cache_max_age_s: float = Field(default=86_400, ge=0)


class AutoConfig(BaseModel):
    """Auto mode (the default): the harness picks the strategy. See auto.decide_mode."""

    model_config = ConfigDict(extra="forbid")

    # Answer tier for a question without sources, and for a stuffed call. Unset: fallback_tier.
    closed_book_tier: str | None = None
    # Sources are stuffed into one call when the estimated prompt is at or under this
    # share of the answering tier's context_window_tokens.
    stuffed_fill_ratio: float = Field(default=0.8, gt=0, le=1)
    # Before the closed-book call, ask one tier whether a short program can compute the
    # answer; a yes sends the question to execute mode. Off until measured.
    # --computation-check / --no-computation-check.
    computation_check: bool = False
    # The tier asked: a chat triage tier or an answer tier. It gets the question text only.
    computation_check_tier: str = TRIAGE_TIER
    # A yes below this confidence counts as no.
    computation_check_threshold: float = Field(default=0.8, ge=0, le=1)
    # When the check says yes, execute mode writes two programs and compares them
    # (the cross-check), whatever execute.cross_check says. False: one program.
    computation_cross_check: bool = True


class ExecuteConfig(BaseModel):
    """Execute mode: the model writes a program, the sandbox runs it, code compares."""

    model_config = ConfigDict(extra="forbid")

    # Answer tier that writes the program and the claimed answer. --tier overrides it.
    tier: str = "mid"
    # Ollama only: the `think` value sent with the draft and repair calls (false, true,
    # low, medium, high). Unset: the tier's own setting. Writing a short program needs
    # little hidden reasoning, and the model otherwise drafts the program twice.
    think: bool | Literal["low", "medium", "high"] | None = None
    # Repair attempts after a program that failed or timed out, with its stderr fed back.
    # In a cross-check each program has this many repairs of its own.
    max_repairs: int = Field(default=1, ge=0, le=1)
    # Write a second program independently from the question and compare the two outputs
    # in code; only matching outputs are `verified`. Off until measured. --cross-check.
    cross_check: bool = False
    # Answer tier that writes program B. Unset: the tier that writes program A, and B's
    # prompt then asks for a different method. A different model is the better choice:
    # two models share fewer failure modes than two drafts from one.
    cross_check_tier: str | None = None
    # Ollama only: `think` for program B's calls. Unset: `think` above when B runs on
    # program A's tier, otherwise the B tier's own setting.
    cross_check_think: bool | Literal["low", "medium", "high"] | None = None
    # Draft and run the two programs at the same time. They share no state.
    cross_check_parallel: bool = True
    # Wall-clock limit for one program, enforced inside the sandbox and again on the host.
    timeout_s: float = Field(default=20, gt=0, le=300)
    # Cap on each of stdout and stderr; a program over it fails with output_too_large.
    max_output_bytes: int = Field(default=20_000, gt=0)
    max_program_bytes: int = Field(default=20_000, gt=0)
    cpus: int = Field(default=2, gt=0)
    # sbx binary units; its minimum is 512m.
    memory: str = "1g"
    # Covers a first run that has to pull the sandbox image.
    create_timeout_s: float = Field(default=180, gt=0)
    sbx_path: str = "sbx"
    # A host the global sbx policy allows. Before a program runs, sbx must report this
    # host as denied for the new sandbox; otherwise the program is not run.
    network_canary: str = "pypi.org:443"


class HarnessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    providers: dict[str, ProviderConfig]
    tiers: dict[str, TierConfig]
    confidence_threshold: float = Field(default=0.80, ge=0, le=1)
    fallback_tier: str = "mid"
    escalation: dict[str, str] = Field(default_factory=dict)
    allow_cloud_fallback: bool = True
    store_prompt_text: bool = True
    triage_max_chars: int = Field(default=4000, gt=0)
    # Which triage tier classifies each question. Must have role: triage.
    triage_tier: str = TRIAGE_TIER
    # Optional. Without it only route mode is available.
    chain: ChainConfig | None = None
    # Optional. Without it the openbook and stuffed modes are not available.
    openbook: OpenbookConfig | None = None
    auto: AutoConfig = Field(default_factory=AutoConfig)
    # Optional. Without it execute mode is not available.
    execute: ExecuteConfig | None = None

    # Set by load_config; not part of the YAML.
    sha256: str = ""
    source_path: str = ""

    @model_validator(mode="after")
    def _check_references(self) -> "HarnessConfig":
        for required_tier in REQUIRED_TIERS:
            if required_tier not in self.tiers:
                raise ValueError(f"missing required tier '{required_tier}'")

        self.tiers[TRIAGE_TIER].role = "triage"
        triage_tiers = set(self.triage_tier_names())
        answer_tiers = set(self.answer_tier_names())

        for tier_name, tier in self.tiers.items():
            if tier.provider not in self.providers:
                raise ValueError(
                    f"tier '{tier_name}' uses unknown provider '{tier.provider}'"
                )
            if self.providers[tier.provider].kind == "openrouter_decisions" and tier.role == "answer":
                raise ValueError(
                    f"tier '{tier_name}' uses a decisions provider and must set role: triage or verifier"
                )
            if tier.on_error is None:
                continue
            if tier.on_error not in self.tiers:
                raise ValueError(
                    f"tier '{tier_name}' on_error points at unknown tier '{tier.on_error}'"
                )
            if tier.on_error == tier_name:
                raise ValueError(f"tier '{tier_name}' on_error points at itself")
            if tier.on_error in triage_tiers:
                raise ValueError(f"tier '{tier_name}' on_error cannot be a triage tier")
            if tier.on_error not in answer_tiers:
                raise ValueError(f"tier '{tier_name}' on_error must be an answer tier")

        if self.triage_tier not in triage_tiers:
            raise ValueError(
                f"triage_tier '{self.triage_tier}' is not a triage tier; choose one of {sorted(triage_tiers)}"
            )

        if self.fallback_tier not in answer_tiers:
            raise ValueError(f"fallback_tier '{self.fallback_tier}' is not an answer tier")

        for from_tier, to_tier in self.escalation.items():
            if from_tier not in self.tiers:
                raise ValueError(f"escalation source '{from_tier}' is not a tier")
            if to_tier not in answer_tiers:
                raise ValueError(f"escalation target '{to_tier}' is not an answer tier")

        if self.chain is not None:
            verifier_tiers = self.verifier_tier_names()
            if self.chain.verifier_tier not in verifier_tiers:
                raise ValueError(
                    f"chain.verifier_tier '{self.chain.verifier_tier}' is not a verifier tier; "
                    f"choose one of {sorted(verifier_tiers)}"
                )
            if self.chain.repair_tier not in answer_tiers:
                raise ValueError(f"chain.repair_tier '{self.chain.repair_tier}' is not an answer tier")
            alt_verifier = self.chain.alt_verifier_tier
            if alt_verifier is not None:
                if alt_verifier not in verifier_tiers:
                    raise ValueError(
                        f"chain.alt_verifier_tier '{alt_verifier}' is not a verifier tier; "
                        f"choose one of {sorted(verifier_tiers)}"
                    )
                if self.tiers[alt_verifier].model == self.tiers[self.chain.verifier_tier].model:
                    raise ValueError(
                        "chain.alt_verifier_tier must run a different model from chain.verifier_tier"
                    )

        if self.openbook is not None:
            self._check_openbook(self.openbook, answer_tiers)
        if self.execute is not None and self.execute.tier not in answer_tiers:
            raise ValueError(f"execute.tier '{self.execute.tier}' is not an answer tier")
        if (
            self.execute is not None
            and self.execute.cross_check_tier is not None
            and self.execute.cross_check_tier not in answer_tiers
        ):
            raise ValueError(
                f"execute.cross_check_tier '{self.execute.cross_check_tier}' is not an answer tier"
            )
        if self.auto.closed_book_tier is not None and self.auto.closed_book_tier not in answer_tiers:
            raise ValueError(f"auto.closed_book_tier '{self.auto.closed_book_tier}' is not an answer tier")
        check_tier = self.auto.computation_check_tier
        if check_tier not in triage_tiers and check_tier not in answer_tiers:
            raise ValueError(
                f"auto.computation_check_tier '{check_tier}' is not a triage tier or an answer tier"
            )
        if self.providers[self.tiers[check_tier].provider].kind == "openrouter_decisions":
            raise ValueError(
                f"auto.computation_check_tier '{check_tier}' uses a decisions provider; it needs a chat model"
            )
        if self.auto.computation_check and self.execute is None:
            raise ValueError("auto.computation_check needs an `execute:` block in the config")
        return self

    def auto_closed_book_tier(self) -> str:
        return self.auto.closed_book_tier or self.fallback_tier

    def _check_openbook(self, openbook: OpenbookConfig, answer_tiers: set[str]) -> None:
        verifier_tiers = self.verifier_tier_names()
        if openbook.drafter_tier not in answer_tiers:
            raise ValueError(f"openbook.drafter_tier '{openbook.drafter_tier}' is not an answer tier")
        if openbook.escalation_tier not in answer_tiers:
            raise ValueError(
                f"openbook.escalation_tier '{openbook.escalation_tier}' is not an answer tier"
            )
        if openbook.grounding_verifier_tier not in verifier_tiers:
            raise ValueError(
                f"openbook.grounding_verifier_tier '{openbook.grounding_verifier_tier}' is not a "
                f"verifier tier; choose one of {sorted(verifier_tiers)}"
            )
        alt_verifier = openbook.alt_grounding_verifier_tier
        if alt_verifier is not None:
            if alt_verifier not in verifier_tiers:
                raise ValueError(
                    f"openbook.alt_grounding_verifier_tier '{alt_verifier}' is not a verifier "
                    f"tier; choose one of {sorted(verifier_tiers)}"
                )
            if self.tiers[alt_verifier].model == self.tiers[openbook.grounding_verifier_tier].model:
                raise ValueError(
                    "openbook.alt_grounding_verifier_tier must run a different model from "
                    "openbook.grounding_verifier_tier"
                )
        embedding_tier = openbook.embedding_tier
        if embedding_tier is not None:
            if embedding_tier not in self.tiers or self.tiers[embedding_tier].role != "embedding":
                raise ValueError(
                    f"openbook.embedding_tier '{embedding_tier}' is not a tier with role: embedding"
                )
            if self.providers[self.tiers[embedding_tier].provider].kind != "ollama_native":
                raise ValueError("openbook.embedding_tier must use an ollama_native provider")

    def triage_tier_names(self) -> list[str]:
        return [name for name, tier in self.tiers.items() if tier.role == "triage"]

    def answer_tier_names(self) -> list[str]:
        return [name for name, tier in self.tiers.items() if tier.role == "answer"]

    def verifier_tier_names(self) -> list[str]:
        return [name for name, tier in self.tiers.items() if tier.role == "verifier"]


MODEL_PLACEHOLDER = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


def unresolved_model_variable(model: str) -> str | None:
    """The variable name when `model` is still a ${NAME} placeholder, else None."""
    placeholder = MODEL_PLACEHOLDER.match(model)
    return placeholder.group(1) if placeholder else None


def resolve_model_placeholders(parsed: dict[str, Any]) -> None:
    """Replace a tier's `model: ${NAME}` with the value of environment variable NAME.

    An unset or empty variable leaves the placeholder in place: the config still loads,
    and a call to that tier fails closed with `missing_model`.
    """
    tiers = parsed.get("tiers")
    if not isinstance(tiers, dict):
        return
    for tier in tiers.values():
        if not isinstance(tier, dict) or not isinstance(tier.get("model"), str):
            continue
        variable_name = unresolved_model_variable(tier["model"])
        if variable_name and os.environ.get(variable_name):
            tier["model"] = os.environ[variable_name]


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> HarnessConfig:
    config_path = Path(path)
    try:
        raw_bytes = config_path.read_bytes()
    except OSError as read_error:
        raise ConfigError(f"cannot read config file {config_path}: {read_error}") from read_error

    try:
        parsed = yaml.safe_load(raw_bytes)
    except yaml.YAMLError as yaml_error:
        raise ConfigError(f"invalid YAML in {config_path}: {yaml_error}") from yaml_error

    if not isinstance(parsed, dict):
        raise ConfigError(f"{config_path} must contain a YAML mapping at the top level")

    resolve_model_placeholders(parsed)
    try:
        config = HarnessConfig.model_validate(parsed)
    except ValidationError as validation_error:
        raise ConfigError(f"invalid config {config_path}:\n{validation_error}") from validation_error

    config.sha256 = hashlib.sha256(raw_bytes).hexdigest()
    config.source_path = str(config_path)
    return config


def load_env_file(path: str | Path = DEFAULT_ENV_PATH) -> list[str]:
    """Load KEY=VALUE lines into os.environ without overriding existing variables.

    Returns the NAMES that were set. Values are never returned or logged.
    A missing file is not an error: the harness runs local-only without one.
    """
    env_path = Path(path)
    if not env_path.is_file():
        return []

    loaded_names: list[str] = []
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        name, _, value = line.partition("=")
        name = name.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if not name or not value or name in os.environ:
            continue
        os.environ[name] = value
        loaded_names.append(name)
    return loaded_names
