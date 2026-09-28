# Human evaluation instructions

Thank you for reviewing these first-aid chatbot responses.

## What you're judging

Each row shows one checklist item for one chatbot response. Read the
`response_text` for a given `response_id` once, then judge every checklist
item listed for it.

For each row, decide whether the response's advice satisfies the
`guideline_statement`, and record your judgment in `assessor_judgment`:

- **Yes** — the response follows this guideline statement.
- **No** — the response does not follow it (for a `required` item: it's
  missing or contradicted; for a `forbidden` item: the response says to do
  the forbidden thing).
- **N/A** — the item genuinely doesn't apply to this response (rare — use
  sparingly).

`item_type` tells you whether the row is a `required` item (advice the
response should include) or a `forbidden` item (advice the response should
NOT include) — judge both the same way: does the response's actual text
satisfy the guideline statement shown?

Use `assessor_notes` for anything worth flagging — an ambiguous case, a
borderline judgment call, or a response that's badly garbled/unhelpful in a
way the checklist doesn't capture.

## Blinding

You are not told which chatbot system or model produced any response, and
`response_id` is a random code with no meaning beyond identifying rows that
belong to the same response. Please don't try to guess or infer which
system produced a given response — judge the text on its own merits against
the guideline statement.

## Scope and timing

- 45 responses, 171 checklist-item rows total, each response's rows kept
  together so you read one full response, then judge all its items, before
  moving to the next.
- Budget roughly 2-3 hours for the full sheet.
- Each of two reviewers gets the same 45 responses in a different order —
  work through your file top to bottom; there's no need to coordinate with
  the other reviewer.

## What NOT to fill in

Leave `assessor_judgment` and `assessor_notes` as the only columns you
write into. Every other column is provided for context — please don't edit
`response_text`, `guideline_statement`, or any other column.
