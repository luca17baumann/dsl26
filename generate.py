#!/usr/bin/env python3
"""Generate responses from WildChat dataset using multiple sglang-served models with adaptation."""

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path
from dotenv import load_dotenv

import aiohttp
import yaml
from datasets import load_dataset
from tqdm.asyncio import tqdm as tqdm_asyncio


# Track if we've printed the first error details (per endpoint type)
_first_responder_error_printed = False
_first_judge_error_printed = False

# Judge prompt for checking if response needs adaptation
JUDGE_CONSISTENCY_TEMPLATE = """You are evaluating whether an AI assistant's response fits well with how the conversation continued.

Conversation context:
{context}

AI Response being evaluated:
{response}

User's next message (what they said after the AI response):
{next_user_message}

Evaluate whether the AI response makes sense given that the user followed up with the above message.
- Check if your response answers the original user request appropriately - top priority metric!
- If the user's follow-up is a natural continuation, the response fits well.
- If the user seems confused, asks for clarification, or the follow-up doesn't connect, the response needs to cause the reason why the user asks it.
- Important! Check the response contains the reason the user replied as they did, for example, the response might have factual errors, be off-topic, or incomplete and cause the user to ask for clarification.

Mark "needs adaptation" if the response needs only MINOR content changes to fit better with the next user message.

Respond with a JSON object:
{{"verdict": "pass" or "needs_adaptation", "reason": "brief explanation"}}

Only output the JSON, nothing else."""

# Prompt for adapting a response to fit better with the next user message
ADAPT_RESPONSE_TEMPLATE = """You previously generated a response that might not completely fit well with how the conversation actually continued. Please adapt your response.

Conversation so far:
{context}

Your original response:
{original_response}

Problem: {reason}

The user's actual next message was:
{next_user_message}

Please generate an adapted response that would naturally lead to the user's next message. The response should:
1. Still address the user's current question/request
2. But be written in a way that makes the user's follow-up message a natural continuation

Output only the adapted response, nothing else."""

# Sanity verification prompt - judge checks if the final response is acceptable
SANITY_VERIFICATION_TEMPLATE = """You are a quality assurance judge evaluating an AI assistant's response.

Conversation context:
{context}

User's question/request:
{user_message}

AI Response:
{response}

Evaluate whether this response is acceptable:
1. Does it address the user's request appropriately?
2. Is it coherent and well-formed?
3. Does it contain any obvious errors, hallucinations, or problematic content?
4. Is it helpful and relevant?

Respond with a JSON object:
{{"verdict": "pass" or "fail", "reason": "brief explanation"}}

Only output the JSON, nothing else."""


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    load_dotenv()  # Load environment variables from .env file if present
    import os
    with open(config_path, "r") as f:
        content = f.read()
        # Simple env var substitution for ${CSCS_SERVING_API} or ENV=CSCS_SERVING_API
        api_key = os.environ.get("CSCS_SERVING_API", "")
        content = content.replace("ENV=CSCS_SERVING_API", f'"{api_key}"')
        return yaml.safe_load(content)


def extract_conversation_turns(conversation: list[dict]) -> list[dict]:
    """Extract user/assistant turn pairs from conversation."""
    turns = []
    for msg in conversation:
        role = msg.get("role")
        content = msg.get("content")
        if role in ("user", "assistant") and content:
            turns.append({"role": role, "content": content})
    return turns


def build_url(base_url: str) -> str:
    """Build the chat completions URL from base URL."""
    base_url = base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    return f"{base_url}/chat/completions"


async def call_api(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    api_config: dict,
    gen_config: dict,
    messages: list[dict],
    error_prefix: str,
    is_judge: bool = False,
) -> tuple[str | None, str | None]:
    """Make an API call and return (response_text, error_msg)."""
    global _first_responder_error_printed, _first_judge_error_printed

    url = build_url(api_config["base_url"])
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_config['api_key']}",
    }
    payload = {
        "model": api_config["model"],
        "messages": messages,
        "max_tokens": gen_config["max_tokens"],
        "temperature": gen_config["temperature"],
        "top_p": gen_config["top_p"],
    }

    first_error_printed = _first_judge_error_printed if is_judge else _first_responder_error_printed

    async with semaphore:
        try:
            async with session.post(url, headers=headers, json=payload) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    response_text = result["choices"][0]["message"]["content"]
                    return response_text, None
                else:
                    error_text = await resp.text()
                    error_msg = f"HTTP {resp.status}: {error_text}"

                    if not first_error_printed:
                        if is_judge:
                            _first_judge_error_printed = True
                        else:
                            _first_responder_error_printed = True
                        print(f"\n[{error_prefix} ERROR] URL: {url}", file=sys.stderr)
                        print(f"[{error_prefix} ERROR] Status: {resp.status}", file=sys.stderr)
                        print(f"[{error_prefix} ERROR] Headers: {dict(resp.headers)}", file=sys.stderr)
                        print(f"[{error_prefix} ERROR] Response body:\n{error_text}", file=sys.stderr)
                        print(f"[{error_prefix} ERROR] Request payload:\n{json.dumps(payload, indent=2)}", file=sys.stderr)

                    return None, error_msg
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"

            if not first_error_printed:
                if is_judge:
                    _first_judge_error_printed = True
                else:
                    _first_responder_error_printed = True
                print(f"\n[{error_prefix} ERROR] URL: {url}", file=sys.stderr)
                print(f"[{error_prefix} ERROR] Exception: {error_msg}", file=sys.stderr)

            return None, error_msg


def format_context(messages: list[dict]) -> str:
    """Format conversation messages for prompts."""
    lines = []
    for msg in messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n\n".join(lines)


def parse_judge_response(judge_response: str | None) -> tuple[str, str]:
    """Parse judge response JSON, return (verdict, reason)."""
    if judge_response is None:
        return "error", "No response received"
    try:
        judge_text = judge_response.strip()
        # Handle potential markdown code blocks
        if judge_text.startswith("```"):
            judge_text = judge_text.split("```")[1]
            if judge_text.startswith("json"):
                judge_text = judge_text[4:]
            judge_text = judge_text.strip()

        judge_result = json.loads(judge_text)
        return judge_result.get("verdict", "unknown"), judge_result.get("reason", "")
    except (json.JSONDecodeError, IndexError):
        return "parse_error", judge_response


async def process_conversation(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    conversation: list[dict],
    config: dict,
    sample_id: int,
) -> dict:
    """Process a multi-turn conversation with same responder for all turns."""
    responders = config["responders"]
    judge_config = config["judge"]
    gen_config = config["generation"]
    judge_gen_config = config["judge_generation"]
    adaptation_gen_config = config.get("adaptation_generation", gen_config)
    max_adaptations = config.get("processing", {}).get("max_adaptations", 2)
    max_retries = config.get("processing", {}).get("max_retries", 3)
    max_conv_turns = config.get("processing", {}).get("max_conv_turns", 0)

    # Extract turns from conversation
    turns = extract_conversation_turns(conversation)

    # Find user turn indices (where we need to generate responses)
    user_turn_indices = [i for i, t in enumerate(turns) if t["role"] == "user"]
    original_num_turns = len(user_turn_indices)

    # Limit conversation turns if max_conv_turns is set (0 = no limit)
    truncated = False
    if max_conv_turns > 0 and len(user_turn_indices) > max_conv_turns:
        user_turn_indices = user_turn_indices[:max_conv_turns]
        truncated = True

    if not user_turn_indices:
        return {
            "id": sample_id,
            "conversation": [],
            "metadata": {
                "num_turns": 0,
                "original_num_turns": original_num_turns,
                "truncated": False,
                "responder": None,
                "responder_model": None,
                "status": "no_user_turns",
            },
        }

    # Randomly sample a responder (same for entire conversation)
    responder = random.choice(responders)

    result = {
        "id": sample_id,
        "conversation": [],  # Will be built in original format
        "metadata": {
            "num_turns": len(user_turn_indices),
            "original_num_turns": original_num_turns,
            "truncated": truncated,
            "responder": responder["name"],
            "responder_model": responder["model"],
            "status": "success",
            "turns_detail": [],  # Detailed info per turn
        },
    }

    # Process each user turn
    generated_history = []  # Track our generated conversation

    for turn_idx, user_idx in enumerate(user_turn_indices):
        is_last_turn = (turn_idx == len(user_turn_indices) - 1)

        # Build messages: all previous generated history + current user message
        user_message = turns[user_idx]["content"]
        messages = generated_history + [{"role": "user", "content": user_message}]

        # Generate initial response with retries
        response_text = None
        response_error = None
        retry_count = 0

        while response_text is None and retry_count < max_retries:
            print(f"[{responder['name']}] Turn {turn_idx}: Generating response (attempt {retry_count + 1})...", flush=True)
            response_text, response_error = await call_api(
                session, semaphore, responder, gen_config, messages,
                error_prefix=f"RESPONDER:{responder['name']}",
                is_judge=False,
            )
            if response_text is None:
                print(f"[{responder['name']}] Error: {response_error}. Retrying...", flush=True)
                retry_count += 1
            else:
                print(f"[{responder['name']}] Turn {turn_idx}: Generation successful! Length: {len(response_text)} chars", flush=True)

        turn_result = {
            "turn": turn_idx,
            "user_message": user_message,
            "original_response": response_text,
            "response_status": "success" if response_text else "error",
            "retry_count": retry_count,
            "adaptations": [],
        }

        if response_error:
            turn_result["response_error"] = response_error
            turn_result["final_response"] = None
            result["metadata"]["turns_detail"].append(turn_result)
            result["metadata"]["status"] = "partial_error"
            # Still add to history and conversation for subsequent turns (with empty response)
            generated_history.append({"role": "user", "content": user_message})
            generated_history.append({"role": "assistant", "content": ""})
            result["conversation"].append({"role": "user", "content": user_message})
            result["conversation"].append({"role": "assistant", "content": ""})
            continue

        final_response = response_text

        # For non-last turns, check consistency and adapt if needed
        if not is_last_turn:
            next_user_idx = user_turn_indices[turn_idx + 1]
            next_user_message = turns[next_user_idx]["content"]
            turn_result["next_user_message"] = next_user_message

            current_response = response_text
            adaptation_count = 0

            while adaptation_count < max_adaptations:
                print(f"[{responder['name']}] Turn {turn_idx}: Running judge consistency check...", flush=True)
                # Check if response fits with next user message
                judge_prompt = JUDGE_CONSISTENCY_TEMPLATE.format(
                    context=format_context(messages),
                    response=current_response,
                    next_user_message=next_user_message,
                )
                judge_messages = [{"role": "user", "content": judge_prompt}]
                # Use responder model for consistency check (self-evaluation)
                judge_response, judge_error = await call_api(
                    session, semaphore, responder, adaptation_gen_config, judge_messages,
                    error_prefix=f"CONSISTENCY:{responder['name']}",
                    is_judge=False,
                )

                if judge_error or judge_response is None:
                    turn_result["adaptations"].append({
                        "attempt": adaptation_count,
                        "judge_status": "error",
                        "judge_error": judge_error or "No response received",
                    })
                    break

                verdict, reason = parse_judge_response(judge_response)

                if verdict == "pass":
                    print(f"[{responder['name']}] Turn {turn_idx}: Consistency check PASSED", flush=True)
                    turn_result["adaptations"].append({
                        "attempt": adaptation_count,
                        "judge_verdict": verdict,
                        "judge_reason": reason,
                    })
                    final_response = current_response
                    break
                elif verdict == "needs_adaptation":
                    print(f"[{responder['name']}] Turn {turn_idx}: Needs adaptation ({reason}). Generating new response...", flush=True)
                    # Ask model to adapt the response
                    adapt_prompt = ADAPT_RESPONSE_TEMPLATE.format(
                        context=format_context(messages),
                        original_response=current_response,
                        reason=reason,
                        next_user_message=next_user_message,
                    )
                    adapt_messages = [{"role": "user", "content": adapt_prompt}]
                    adapted_response, adapt_error = await call_api(
                        session, semaphore, responder, adaptation_gen_config, adapt_messages,
                        error_prefix=f"ADAPT:{responder['name']}",
                        is_judge=False,
                    )

                    turn_result["adaptations"].append({
                        "attempt": adaptation_count,
                        "judge_verdict": verdict,
                        "judge_reason": reason,
                        "adapted_response": adapted_response,
                        "adapt_status": "success" if adapted_response else "error",
                        "adapt_error": adapt_error,
                    })

                    if adapted_response:
                        current_response = adapted_response
                        final_response = adapted_response
                    else:
                        break

                    adaptation_count += 1
                else:
                    # Parse error or unknown verdict
                    turn_result["adaptations"].append({
                        "attempt": adaptation_count,
                        "judge_verdict": verdict,
                        "judge_reason": reason,
                    })
                    break

            turn_result["num_adaptations"] = adaptation_count
            turn_result["final_verdict"] = verdict if 'verdict' in dir() else "unknown"

        turn_result["final_response"] = final_response

        # Sanity verification by judge model after adaptation step
        print(f"[{responder['name']}] Turn {turn_idx}: Running final sanity verification...", flush=True)
        sanity_prompt = SANITY_VERIFICATION_TEMPLATE.format(
            context=format_context(generated_history) if generated_history else "Start of conversation",
            user_message=user_message,
            response=final_response,
        )
        sanity_messages = [{"role": "user", "content": sanity_prompt}]
        sanity_response, sanity_error = await call_api(
            session, semaphore, judge_config, judge_gen_config, sanity_messages,
            error_prefix="SANITY_JUDGE",
            is_judge=True,
        )

        if sanity_error or sanity_response is None:
            turn_result["sanity_check"] = {
                "status": "error",
                "error": sanity_error or "No response received",
                "verdict": "error",
            }
        else:
            sanity_verdict, sanity_reason = parse_judge_response(sanity_response)
            print(f"[{responder['name']}] Turn {turn_idx}: Sanity verification verdict: {sanity_verdict}", flush=True)
            turn_result["sanity_check"] = {
                "status": "success",
                "verdict": sanity_verdict,
                "reason": sanity_reason,
            }

        # Update generated history with final response
        generated_history.append({"role": "user", "content": user_message})
        generated_history.append({"role": "assistant", "content": final_response})

        # Add to conversation in original format
        result["conversation"].append({"role": "user", "content": user_message})
        result["conversation"].append({"role": "assistant", "content": final_response})

        result["metadata"]["turns_detail"].append(turn_result)

    # Aggregate sanity verdicts from all turns
    sanity_verdicts = []
    for turn_detail in result["metadata"]["turns_detail"]:
        sanity_check = turn_detail.get("sanity_check", {})
        verdict = sanity_check.get("verdict", "unknown")
        sanity_verdicts.append(verdict)

    # Conversation passes only if all turns pass sanity check
    all_passed = all(v == "pass" for v in sanity_verdicts)
    result["metadata"]["sanity_verdicts"] = sanity_verdicts
    result["metadata"]["sanity_pass"] = all_passed

    return result


def save_checkpoint(results: list[dict], output_path: Path) -> None:
    """Save checkpoint to JSONL file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


async def process_dataset(config: dict) -> list[dict]:
    """Process the dataset and generate responses."""
    dataset_config = config["dataset"]
    processing_config = config["processing"]
    output_config = config["output"]

    # Set random seed
    seed = processing_config["seed"]
    random.seed(seed)

    # Get checkpoint settings
    save_every = output_config.get("save_every", 100)
    output_path = Path(output_config["path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print(f"Loading dataset: {dataset_config['name']}...")
    dataset = load_dataset(
        dataset_config["name"],
        split=dataset_config["split"],
        trust_remote_code=True,
    )

    # Sample from dataset
    num_samples = min(dataset_config["num_samples"], len(dataset))
    indices = random.sample(range(len(dataset)), num_samples)

    # Extract conversations
    conversations = []
    for idx in indices:
        conversation = dataset[idx].get("conversation", [])
        if conversation:
            conversations.append((idx, conversation))

    print(f"Extracted {len(conversations)} valid conversations from {num_samples} samples")

    # Count total turns for progress reporting
    total_turns = sum(
        len([t for t in extract_conversation_turns(conv) if t["role"] == "user"])
        for _, conv in conversations
    )
    print(f"Total user turns to process: {total_turns}")

    # Setup async HTTP session and semaphore for concurrency control
    concurrency = processing_config["concurrency"]
    semaphore = asyncio.Semaphore(concurrency)

    connector = aiohttp.TCPConnector(limit=concurrency * 2)
    timeout = aiohttp.ClientTimeout(total=600)  # Increased timeout from 120 to 600 seconds

    results = []
    last_checkpoint = 0

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [
            asyncio.create_task(process_conversation(session, semaphore, conv, config, idx))
            for idx, conv in conversations
        ]

        with tqdm_asyncio(total=len(tasks), desc="Processing conversations") as pbar:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)
                pbar.update(1)

                # Save checkpoint every save_every results
                if len(results) - last_checkpoint >= save_every:
                    save_checkpoint(results, output_path)
                    last_checkpoint = len(results)
                    pbar.set_postfix({"checkpointed": len(results)})

    # Final save
    save_checkpoint(results, output_path)

    return results


def print_stats(results: list[dict], output_path: str) -> None:
    """Print statistics about the results."""
    output_path = Path(output_path)

    # Helper to get metadata
    def get_meta(r):
        return r.get("metadata", {})

    def get_turns(r):
        return get_meta(r).get("turns_detail", [])

    # Compute stats
    total_conversations = len(results)
    total_turns = sum(len(get_turns(r)) for r in results)

    response_success = sum(
        1 for r in results for t in get_turns(r)
        if t.get("response_status") == "success"
    )
    response_error = sum(
        1 for r in results for t in get_turns(r)
        if t.get("response_status") == "error"
    )

    # Adaptation stats
    total_adaptations = sum(
        t.get("num_adaptations", 0)
        for r in results for t in get_turns(r)
    )
    turns_needing_adaptation = sum(
        1 for r in results for t in get_turns(r)
        if t.get("num_adaptations", 0) > 0
    )

    # Retry stats
    total_retries = sum(
        t.get("retry_count", 0)
        for r in results for t in get_turns(r)
    )
    turns_needing_retries = sum(
        1 for r in results for t in get_turns(r)
        if t.get("retry_count", 0) > 0
    )

    # Truncation stats
    truncated_conversations = sum(1 for r in results if get_meta(r).get("truncated", False))

    # Sanity check stats
    sanity_pass_conversations = sum(1 for r in results if get_meta(r).get("sanity_pass", False))
    sanity_fail_conversations = total_conversations - sanity_pass_conversations

    # Per-turn sanity stats
    sanity_pass_turns = sum(
        1 for r in results for t in get_turns(r)
        if t.get("sanity_check", {}).get("verdict") == "pass"
    )
    sanity_fail_turns = sum(
        1 for r in results for t in get_turns(r)
        if t.get("sanity_check", {}).get("verdict") == "fail"
    )
    sanity_error_turns = sum(
        1 for r in results for t in get_turns(r)
        if t.get("sanity_check", {}).get("verdict") in ("error", "parse_error", "unknown")
    )

    # Count per responder
    responder_counts = {}
    for r in results:
        name = get_meta(r).get("responder", "unknown")
        if name:
            responder_counts[name] = responder_counts.get(name, 0) + 1

    # Turn distribution
    turn_counts = {}
    for r in results:
        num_turns = get_meta(r).get("num_turns", 0)
        turn_counts[num_turns] = turn_counts.get(num_turns, 0) + 1

    print(f"\nResults saved to: {output_path}")
    print(f"\nConversations: {total_conversations}, Total turns: {total_turns}")
    print(f"Truncated conversations: {truncated_conversations}")
    print(f"Response stats: {response_success} success, {response_error} errors")
    print(f"Retry stats: {turns_needing_retries} turns retried, {total_retries} total retries")
    print(f"Adaptation stats: {turns_needing_adaptation} turns adapted, {total_adaptations} total adaptation attempts")
    print(f"Sanity check stats (conversations): {sanity_pass_conversations} pass, {sanity_fail_conversations} fail")
    print(f"Sanity check stats (turns): {sanity_pass_turns} pass, {sanity_fail_turns} fail, {sanity_error_turns} error/unknown")
    print(f"\nResponder distribution:")
    for name, count in sorted(responder_counts.items()):
        print(f"  {name}: {count} conversations")
    print(f"\nTurns per conversation:")
    for num_turns, count in sorted(turn_counts.items()):
        print(f"  {num_turns} turns: {count} conversations")


def main():
    parser = argparse.ArgumentParser(
        description="Generate responses from WildChat dataset with multiple models and adaptation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        help="Override concurrency from config",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        help="Override num_samples from config",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Override seed from config",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Override output path from config",
    )
    parser.add_argument(
        "--max-adaptations",
        type=int,
        help="Max adaptation attempts per turn (default: 2)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        help="Max retries if response generation fails (default: 3)",
    )
    parser.add_argument(
        "--max-conv-turns",
        type=int,
        help="Max conversation turns to process, 0 = no limit (default: 0)",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Apply CLI overrides
    if args.concurrency:
        config["processing"]["concurrency"] = args.concurrency
    if args.num_samples:
        config["dataset"]["num_samples"] = args.num_samples
    if args.seed:
        config["processing"]["seed"] = args.seed
    if args.output:
        config["output"]["path"] = args.output
    if args.max_adaptations:
        config["processing"]["max_adaptations"] = args.max_adaptations
    if args.max_retries:
        config["processing"]["max_retries"] = args.max_retries
    if args.max_conv_turns is not None:
        config["processing"]["max_conv_turns"] = args.max_conv_turns

    max_adaptations = config.get("processing", {}).get("max_adaptations", 2)
    max_retries = config.get("processing", {}).get("max_retries", 3)
    max_conv_turns = config.get("processing", {}).get("max_conv_turns", 0)

    # Print config summary
    print("Configuration:")
    print(f"  Responders:")
    for r in config["responders"]:
        print(f"    - {r['name']}: {r['model']} @ {r['base_url']}")
    print(f"  Judge: {config['judge']['model']} @ {config['judge']['base_url']}")
    print(f"  Concurrency: {config['processing']['concurrency']}")
    print(f"  Num samples: {config['dataset']['num_samples']}")
    print(f"  Seed: {config['processing']['seed']}")
    print(f"  Max adaptations: {max_adaptations}")
    print(f"  Max retries: {max_retries}")
    print(f"  Max conv turns: {max_conv_turns if max_conv_turns > 0 else 'no limit'}")
    print(f"  Output: {config['output']['path']}")
    print(f"  Save every: {config['output'].get('save_every', 100)} conversations")
    print()

    # Run async processing
    results = asyncio.run(process_dataset(config))

    # Print stats (results already saved via checkpointing)
    print_stats(results, config["output"]["path"])


if __name__ == "__main__":
    main()
