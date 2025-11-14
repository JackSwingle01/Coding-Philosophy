
## Core Principles - Philosophy
*Why We Code This Way*

1. **Reduce Cognitive Load** - Code should be immediately comprehensible. Don't make future-you load database schemas, trace abstractions, or hold complex state in working memory. Favor clarity over cleverness.

2. **Deep Modules with Simple Interfaces** - Create abstractions that hide complexity behind clear, stable interfaces. When calling `Database.get_connection()` (see `db.py`), you shouldn't need to know connection pooling internals, min/max connections, or PostgreSQL configuration. Pull complexity downward into implementations, keep call sites simple and focused on intent.

3. **Explicit Over Clever** - Direct control flow beats "clean" abstractions that hide behavior. If/elif chains are fine if they're clear. Avoid reflection, magic, or patterns that require tracing through multiple files to understand.

4. **Single Source of Truth (SSOT)** - Each piece of knowledge has exactly one authoritative source. One way to get a user, one place for rendering logic, one file for configuration. Don't repeat yourself (DRY) - duplication creates maintenance burden and inconsistency. When you need to change how something works, there should be exactly one place to make that change.

5. **Fail Early at Boundaries** - Validate inputs at function entry, raise clear exceptions for invalid state. Don't scatter defensive None checks throughout code. Recover gracefully only at UI/API boundaries where users can respond.

6. **Predictability** - Function signatures and names telegraph behavior. No surprises, no magic side effects, no hidden retries. Code does what it looks like it does. When appropriate, design operations to be idempotent - safe to repeat with the same result.

7. **Self-Documenting** - Type hints + clear naming + explicit contracts > comments explaining confusing code. If it needs extensive comments, consider refactoring.

8. **Simplicity** - Build what's needed now, not what might be needed later (YAGNI). Abstract when duplication creates real maintenance burden, not for theoretical purity. Wait for 3 real examples before creating an abstraction. Premature optimization and premature abstraction both add complexity without benefit.

9. **Production-First** - Code runs reliably at 2am with 250 users, doesn't leak secrets, handles errors gracefully, and provides observability when things break. Design for the failure modes, not just the happy path.
