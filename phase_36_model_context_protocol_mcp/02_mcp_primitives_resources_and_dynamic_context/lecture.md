# Topic 02: MCP Primitives — Resources & Dynamic Context

---

## 1. The 12-Year-Old Intuitive Analogy: The Magical Library Barcode Scanner

Imagine you are studying in a massive public library that holds millions of books, magazines, old parchment maps, and live news ticker tapes. 

If you had to carry every single book in the library in your backpack at all times just in case you needed it for your homework, your backpack would weigh twenty thousand pounds, collapse your desk, and you wouldn't even be able to open your notebook. That is what happened when developers tried to stuff entire codebases or databases into an AI's context window!

Instead, the library assigns every single document a unique **barcode** (a URI):
- A classic book has a code like `file:///shelf-4/encyclopedia.txt`.
- A database record has a code like `db://students/1042/grades`.
- A live stock ticker has a code like `custom://telemetry/live_prices`.

When your AI assistant needs information, it doesn't carry the whole library. It simply scans the barcode catalog (`resources/list`), reads the specific document it needs into memory (`resources/read`), and if a live document changes, the librarian rings a bell (`notifications/resources/updated`) so the assistant can refresh its notes.

In the Model Context Protocol, **Resources** are these standardized, read-only, URI-addressable data documents.

---

## 2. Complete Mathematical & Protocol Symbol Glossary

| Symbol / Term | Formal Protocol Signature | Description / Role |
| :--- | :--- | :--- |
| $\mathcal{U}$ | $\text{URI} \in \Sigma^*$ | Uniform Resource Identifier identifying a unique data entity (RFC 3986). |
| $\mathcal{T}_{\text{URI}}$ | RFC 6570 URI Template | Parameterized URI schema with expansion placeholders, e.g. `db://users/{id}/profile`. |
| $\mathcal{M}_{\text{mime}}$ | String (e.g. `text/plain`, `application/json`) | IANA MIME type specifying data encoding and formatting. |
| $\text{resources/list}$ | $\mathcal{P} \to \{\text{resources}: [\mathcal{R}], \text{nextCursor}?\}$ | Host queries available static resources with optional cursor pagination. |
| $\text{resources/read}$ | $\{\text{uri}: \mathcal{U}\} \to \{\text{contents}: [\mathcal{C}]\}$ | Fetches contents of resource $\mathcal{U}$ as UTF-8 `text` or Base64 `blob`. |
| $\text{resources/subscribe}$ | $\{\text{uri}: \mathcal{U}\} \to \{\}$ | Registers host client interest in push notifications when $\mathcal{U}$ mutates. |
| $\mathcal{N}_{\text{updated}}$ | `notifications/resources/updated` | Server pushes notification to subscribed clients when resource contents change. |
| $\mathcal{N}_{\text{list\_changed}}$ | `notifications/resources/list_changed` | Server signals that new resources were added or removed from its catalog. |
| $\mathcal{K}_{\text{ctx}}$ | $\mathcal{M}_{\text{messages}}$ | The tokenized context window payload injected into the host LLM prompt. |

---

## 3. The Resource Architecture & Data Model

Resources represent passive, read-only context. Unlike Tools (which execute code and cause side-effects), Resources **only supply information** to the host.

```
┌────────────────────────────────────────────────────────────────────────┐
│                              HOST CLIENT                               │
│                                                                        │
│  1. Discover Catalog:        resources/list                            │
│  2. Read Document:           resources/read {uri: "file:///..."}       │
│  3. Subscribe to Changes:    resources/subscribe {uri: "file:///..."}  │
│  4. Receive Mutation Alert:  notifications/resources/updated           │
└─────────────────────────────────┬──────────────────────────────────────┘
                                  │ JSON-RPC 2.0
                                  ▼
┌────────────────────────────────────────────────────────────────────────┐
│                              MCP SERVER                                │
│                                                                        │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │ Resource Registry:                                             │   │
│   │  • Static Resources:   file:///workspace/README.md             │   │
│   │  • Dynamic Templates:  db://orders/{order_id}/summary          │   │
│   │  • Binary Assets:      custom://assets/diagram.png (base64)    │   │
│   └────────────────────────────────────────────────────────────────┘   │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │ Subscription Table:                                            │   │
│   │  • URI -> Set(Client_IDs)                                      │   │
│   └────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────┘
```

### 3.1. Content Representation: Text vs. Blob
When a host reads a resource via `resources/read`, the server returns a `contents` array:
1. **Text Resources** (UTF-8 encoded files, JSON objects, markdown):
   ```json
   {
     "uri": "file:///workspace/package.json",
     "mimeType": "application/json",
     "text": "{\n  \"name\": \"app\",\n  \"version\": \"1.0.0\"\n}"
   }
   ```
2. **Binary Resources** (PNG images, PDF documents, compiled artifacts):
   ```json
   {
     "uri": "file:///workspace/diagram.png",
     "mimeType": "image/png",
     "blob": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
   }
   ```

---

## 4. RFC 6570 Resource Templates & Variable Expansion

Static listings are sufficient for a fixed folder of files, but impractical for databases with millions of rows. MCP introduces **Resource Templates** adhering to RFC 6570:

$$\mathcal{T}_{\text{URI}} = \text{"postgres://customers/\{customer\_id\}/orders/\{order\_id\}"}$$

When a client queries `resources/templates/list`:
```json
{
  "jsonrpc": "2.0",
  "id": 10,
  "result": {
    "resourceTemplates": [
      {
        "uriTemplate": "db://customers/{customer_id}/orders",
        "name": "Customer Order History",
        "description": "Queries historical purchases for a given customer ID",
        "mimeType": "application/json"
      }
    ]
  }
}
```

The host or LLM fills the placeholder variables:
$$\text{URI} = \text{Expand}\left(\mathcal{T}, \{\text{customer\_id}: 4096\}\right) \to \text{"db://customers/4096/orders"}$$
The client then calls `resources/read` with the expanded URI.

---

## 5. Dynamic Subscriptions & Push Notifications

In standard API architectures, clients must repeatedly poll servers to detect file edits or database mutations. MCP provides a built-in reactive pub-sub protocol:

```
Host Client                                            MCP Server
    │                                                      │
    │  ─── resources/subscribe {"uri": "file:///a.py"} ──►│ (Registers subscription)
    │◄─── {"jsonrpc": "2.0", "id": 1, "result": {}} ──────│
    │                                                      │
    │                                              [File 'a.py' is edited
    │                                               by external process]
    │                                                      │
    │◄─── notifications/resources/updated ─────────────────┤
    │       {"uri": "file:///a.py"}                        │ (Push notification)
    │                                                      │
    │  ─── resources/read {"uri": "file:///a.py"} ────────►│ (Host re-fetches context)
    │◄─── {"result": {"contents": [{"text": "new code"}]}}─┤
```

State invariant for the subscription registry:
$$\mathcal{S}_{\text{registry}} : \mathcal{U} \to \mathcal{P}(\mathcal{C}_{\text{clients}})$$
Upon mutation of $\mathcal{U}$:
$$\forall c \in \mathcal{S}_{\text{registry}}(\mathcal{U}), \quad \text{emit}\left(c, \text{notifications/resources/updated}, \{\text{uri}: \mathcal{U}\}\right)$$

---

## 6. Context Window Injection Strategies

Once resources are fetched, the host must incorporate them into the model's context window without inducing prompt injection or token bloat.

### Recommended Delimited Context Framing:
```text
System: You are an AI assistant. The following authoritative resource documents
have been retrieved from MCP servers to provide grounded factual context.

--- BEGIN RESOURCE: file:///workspace/src/auth.py (MIME: text/x-python) ---
def verify_jwt(token: str) -> bool:
    # Authenticates user signature
    return True
--- END RESOURCE: file:///workspace/src/auth.py ---

--- BEGIN RESOURCE: db://users/402/profile (MIME: application/json) ---
{
  "user_id": 402,
  "role": "Administrator"
}
--- END RESOURCE: db://users/402/profile ---

User: Does user 402 have admin rights according to the system configuration?
```

### Context Management Guidelines:
1. **Explicit Boundaries**: Always bracket resource text with strict, unambiguous delimiters (`--- BEGIN RESOURCE ... ---`) to prevent the LLM from misinterpreting document content as system instructions.
2. **Selective Hydration**: Do not inject all resources at session start. Load metadata (`resources/list`) first; only read and inject resource bodies when requested by user selection or model tool reasoning.
3. **MIME Discrimination**: Deliver text directly; for image/audio blobs, route them through multimodal model vision/audio blocks rather than embedding raw base64 text into the prompt.
