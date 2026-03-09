"""Property-based test for prompt construction.

Feature: nl-search-chatbot, Property 3: Prompt construction includes context and history

Validates: Requirements 4.1, 5.2

For any set of retrieved document chunks and any conversation history, the
constructed LLM prompt must contain all provided chunk texts and all
conversation history messages in chronological order.
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from services.llm import LLMService
from services.search import SearchResultItem
from services.session import ConversationMessage


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_text_st = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "Zs")),
    min_size=1,
    max_size=80,
)

_score_st = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)

_chunk_st = st.builds(
    SearchResultItem,
    relevance_score=_score_st,
    document_id=_text_st,
    document_name=_text_st,
    chunk_text=_text_st,
    s3_uri=_text_st,
)

_role_st = st.sampled_from(["user", "assistant"])

_message_st = st.builds(
    ConversationMessage,
    role=_role_st,
    content=_text_st,
)

_query_st = _text_st


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(
    query=_query_st,
    chunks=st.lists(_chunk_st, min_size=1, max_size=10),
    history=st.lists(_message_st, min_size=0, max_size=10),
)
def test_prompt_contains_all_chunk_texts(
    query: str,
    chunks: list[SearchResultItem],
    history: list[ConversationMessage],
) -> None:
    """**Validates: Requirements 4.1, 5.2**

    Every chunk_text from the provided context chunks must appear in the
    constructed prompt.
    """
    prompt = LLMService.build_prompt(query, chunks, history)

    for chunk in chunks:
        assert chunk.chunk_text in prompt, (
            f"Chunk text {chunk.chunk_text!r} not found in prompt"
        )


@settings(max_examples=100)
@given(
    query=_query_st,
    chunks=st.lists(_chunk_st, min_size=1, max_size=10),
    history=st.lists(_message_st, min_size=0, max_size=10),
)
def test_prompt_contains_all_history_messages(
    query: str,
    chunks: list[SearchResultItem],
    history: list[ConversationMessage],
) -> None:
    """**Validates: Requirements 4.1, 5.2**

    Every conversation history message content must appear in the prompt.
    """
    prompt = LLMService.build_prompt(query, chunks, history)

    for msg in history:
        assert msg.content in prompt, (
            f"History message {msg.content!r} not found in prompt"
        )


@settings(max_examples=100)
@given(
    query=_query_st,
    chunks=st.lists(_chunk_st, min_size=1, max_size=10),
    history=st.lists(_message_st, min_size=0, max_size=10),
)
def test_prompt_contains_query(
    query: str,
    chunks: list[SearchResultItem],
    history: list[ConversationMessage],
) -> None:
    """**Validates: Requirements 4.1, 5.2**

    The user query must appear in the constructed prompt.
    """
    prompt = LLMService.build_prompt(query, chunks, history)

    assert query in prompt, f"Query {query!r} not found in prompt"


@settings(max_examples=100)
@given(
    query=_query_st,
    chunks=st.lists(_chunk_st, min_size=1, max_size=10),
    history=st.lists(_message_st, min_size=2, max_size=10),
)
def test_history_messages_in_chronological_order(
    query: str,
    chunks: list[SearchResultItem],
    history: list[ConversationMessage],
) -> None:
    """**Validates: Requirements 4.1, 5.2**

    History messages must appear in the prompt in the same chronological
    order they were provided — each message appears after the previous one.
    """
    prompt = LLMService.build_prompt(query, chunks, history)

    last_pos = -1
    for msg in history:
        formatted = f"{msg.role}: {msg.content}"
        pos = prompt.find(formatted, last_pos + 1)
        assert pos > last_pos, (
            f"History message {formatted!r} not found after position {last_pos} "
            f"in prompt — chronological order violated"
        )
        last_pos = pos
