# Jack's Software Engineering Principles

**Version:** 1.1
**Purpose:** Universal principles that guide all architectural and implementation decisions across projects.

---

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

These principles define how and why we build software. They apply regardless of language, framework, or project scale.

---

## Details

### 1. Reduce Cognitive Load

Code should be immediately comprehensible. Don't make future-you load database schemas, trace abstractions, or hold complex state in working memory. Favor clarity over cleverness.

**Why:** 6 months from now, you won't remember the context. Code should minimize what you need to understand to make changes safely.

**Example:**
```python
# High cognitive load - need to know schema
cursor.execute("""
    SELECT u.*, p.preferences, o.last_order 
    FROM users u 
    LEFT JOIN user_prefs p ON u.id = p.user_id 
    LEFT JOIN orders o ON u.id = o.user_id 
    WHERE u.id = ?
""", (user_id,))

# Low cognitive load - clear intent
user = user_repo.get_by_id(user_id)
```

### 2. Deep Modules with Simple Interfaces

Create abstractions that hide complexity behind clear, stable interfaces. When calling `user_repo.get_by_id(user_id)`, you shouldn't need to know database schemas, join logic, or caching strategies. Pull complexity downward into implementations, keep call sites simple and focused on intent.

**Why:** The abstraction isn't for flexibility - it's so you can work at the right level of detail for the task at hand. Business logic shouldn't be cluttered with database details.

**Example:**
```python
# Simple interface - hides complexity
class UserRepository:
    def get_by_id(self, user_id: int) -> User:
        # Complex implementation:
        # - Connection pooling
        # - Retry logic
        # - Caching
        # - Error handling
        # - Query optimization
        pass

# Business logic stays clean
def process_order(order_id: int):
    user = user_repo.get_by_id(order_id)  # Simple call, no schema knowledge needed
    if user.is_premium():
        apply_discount(user)
```

### 3. Explicit Over Clever

Direct control flow beats "clean" abstractions that hide behavior. If/elif chains are fine if they're clear. Avoid reflection, magic, or patterns that require tracing through multiple files to understand what actually happens.

**Why:** Debugging and maintenance require understanding control flow. Clever abstractions that save a few lines of code cost hours of confusion later.

**Example:**
```python
# ❌ Clever - uses reflection
def render_block(block: Block):
    method_name = f"_render_{block.block_type}"
    method = getattr(self, method_name)
    return method(block)

# ✅ Explicit - obvious control flow
def render_block(block: Block):
    if block.block_type == "text":
        return self._render_text(block)
    elif block.block_type == "image":
        return self._render_image(block)
    elif block.block_type == "code":
        return self._render_code(block)
    else:
        raise ValueError(f"Unknown block type: {block.block_type}")
```

### 4. Single Source of Truth (SSOT)

Each piece of knowledge has exactly one authoritative source. One way to get a user, one place for configuration, one definition of business rules. Duplication creates maintenance burden and inconsistency.

**Why:** When you need to change how users are fetched, there should be exactly one place to make that change. Scattered duplication means bugs when you forget to update all copies.

**Example:**
```python
# ❌ Multiple sources of truth
# file1.py
max_file_size = 50 * 1024 * 1024

# file2.py  
MAX_SIZE = 52428800

# file3.py
if file.size > 50000000:
    raise Error()

# ✅ Single source of truth
# config.py
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024

# All files import from config
from config import MAX_FILE_SIZE_BYTES
```

### 5. Fail Early at Boundaries

Validate inputs at function entry, raise clear exceptions for invalid state. Don't scatter defensive None checks throughout code. Recover gracefully only at UI/API boundaries where users can respond.

**Why:** Bugs should be obvious during development, not silent data corruption in production. If something is impossible, crash loudly with a clear message at the source.

**Example:**
```python
# ❌ Defensive programming everywhere
def calculate_total(items):
    if items is None:
        return 0
    total = 0
    for item in items:
        if item is not None and item.price is not None:
            total += item.price
    return total

# ✅ Fail early at boundary
def calculate_total(items: list[Item]) -> float:
    if not items:
        raise ValueError("Cannot calculate total for empty order")
    
    return sum(item.price for item in items)  # No None checks needed
```

### 6. Predictability

Function signatures and names telegraph behavior. No surprises, no magic side effects, no hidden retries. Code does what it looks like it does.

**Why:** You should be able to read a function call and know what it does without checking the implementation. Surprise behavior breaks assumptions and causes bugs.

**Example:**
```python
# ❌ Unpredictable - hidden side effects
def get_user(user_id: int) -> User:
    user = db.query(user_id)
    send_analytics("user_fetched")  # Surprise!
    cache.set(user_id, user)  # Surprise!
    return user

# ✅ Predictable - does what it says
def get_user(user_id: int) -> User:
    return db.query(user_id)

def track_user_access(user_id: int) -> User:
    user = get_user(user_id)
    analytics.track("user_accessed", user_id)
    return user
```

### 7. Self-Documenting

Type hints + clear naming + explicit contracts > comments explaining confusing code. If it needs extensive comments, consider refactoring. Code should explain itself through structure.

**Why:** Comments drift from code. Names and types can't lie. Well-structured code tells you what it does; comments tell you why only when necessary.

**Example:**
```python
# ❌ Needs comments to understand
def proc(d, c):
    # Check if company is valid
    if not is_valid(c):
        return None
    # Extract text and chunk it
    t = extract(d)
    chunks = chunk(t, 1000, 200)
    return save(c, d.name, chunks)

# ✅ Self-documenting
def upload_document(file: UploadFile, company_id: int) -> Document:
    validate_company_is_active(company_id)
    text = extract_text_from_file(file)
    chunks = create_text_chunks(text, chunk_size=1000, overlap=200)
    return save_document_with_chunks(company_id, file.filename, chunks)
```

### 8. Simplicity

Build what's needed now, not what might be needed later (YAGNI). Abstract when duplication creates real maintenance burden, not for theoretical purity. Wait for 3 real examples before creating an abstraction. Premature optimization and premature abstraction both add complexity without benefit.

**Why:** Speculative features and early abstractions make code harder to change when requirements actually emerge. Simple, concrete code is easier to understand and modify than clever, "flexible" code built for imaginary futures.

**Example:**
```python
# ❌ Complex - built for imagined future needs
class DataProcessor:
    def __init__(self, strategy: ProcessingStrategy, 
                 cache: CacheProvider,
                 validator: Validator):
        self.strategy = strategy
        self.cache = cache
        self.validator = validator
    
    def process(self, data: Any) -> Any:
        # Built a whole plugin system for one use case
        validated = self.validator.validate(data)
        cached = self.cache.get(validated)
        if cached:
            return cached
        result = self.strategy.execute(validated)
        self.cache.set(validated, result)
        return result

# ✅ Simple - built for actual current need
def process_user_data(data: dict[str, Any]) -> User:
    if "email" not in data or "name" not in data:
        raise ValidationError("Missing required fields")
    
    return User(
        email=data["email"].lower(),
        name=data["name"].strip()
    )

# When you have 3+ similar processing functions, THEN abstract:
def process_data(data: dict, required_fields: list[str], 
                 transform: Callable[[dict], T]) -> T:
    for field in required_fields:
        if field not in data:
            raise ValidationError(f"Missing required field: {field}")
    return transform(data)
```

### 9. Production-First

Code runs reliably at 2am with real users, doesn't leak secrets, handles errors gracefully, and provides observability when things break. Design for the failure modes, not just the happy path.

**Why:** Internal tools and prototypes become production systems. Build with production in mind from the start - it's harder to retrofit reliability later.

**Example:**
```python
# ❌ Development-only thinking
def process_payment(amount: float):
    api_key = "sk_test_12345"  # Hardcoded
    result = payment_api.charge(amount)  # No error handling
    return result

# ✅ Production-first
def process_payment(amount: float) -> PaymentResult:
    api_key = config.PAYMENT_API_KEY  # From env
    
    try:
        result = payment_api.charge(amount, timeout=30)
        logger.info(f"Payment processed: ${amount}")
        return result
        
    except PaymentAPIError as e:
        logger.error(f"Payment failed: ${amount}", exc_info=True)
        raise PaymentError(f"Payment processing failed: {e}") from e
        
    except Timeout:
        logger.error(f"Payment timeout: ${amount}")
        raise PaymentError("Payment service unavailable")
```

---

## Technical Standards

These technical practices implement the philosophy above.

### Type Everything

All function parameters and return types must have type hints. Use built-in types (`list`, `dict`) not typing module (`List`, `Dict`) in Python 3.9+.

**Why:** Types are executable documentation that catches errors early and makes code self-documenting.
```python
# ✅ Good
def process_items(items: list[str], threshold: int) -> dict[str, int]:
    return {item: len(item) for item in items if len(item) > threshold}

# ❌ Bad
def process_items(items, threshold):
    return {item: len(item) for item in items if len(item) > threshold}
```

### Single Return Type - No Unions

Functions return one type or raise exceptions. No unions (`str | None`, `Result | Error`). Use `Optional[T]` only for genuinely nullable values.

**Why:** Union return types force callers to check types and handle multiple cases. Exceptions separate happy path from error handling.
```python
# ❌ Bad - union return type
def get_user(user_id: int) -> User | None:
    return user_repo.find(user_id)

# Caller has to check
user = get_user(123)
if user is None:
    # handle missing user
else:
    # use user

# ✅ Good - raise exception
def get_user(user_id: int) -> User:
    user = user_repo.find(user_id)
    if user is None:
        raise UserNotFoundError(f"User {user_id} not found")
    return user

# Caller focuses on happy path
try:
    user = get_user(123)
    process(user)
except UserNotFoundError:
    # handle error
```

### Complex Returns = Dataclasses

If a return value needs documentation about its fields, create a dataclass instead of returning a dict.

**Why:** Dataclasses are self-documenting, type-safe, and IDE-friendly. Dicts require documentation and offer no type safety.
```python
# ❌ Bad - unclear return type
def get_stats(user_id: int) -> dict:
    """Returns dict with keys: total_orders, revenue, avg_order_value"""
    return {"total_orders": 10, "revenue": 1000.0, "avg_order_value": 100.0}

# ✅ Good - clear dataclass
from dataclasses import dataclass

@dataclass
class UserStats:
    total_orders: int
    revenue: float
    avg_order_value: float

def get_stats(user_id: int) -> UserStats:
    return UserStats(total_orders=10, revenue=1000.0, avg_order_value=100.0)
```

### Minimize Comments

Code should be self-documenting through clear naming and type hints. Only add comments for complex logic that can't be made obvious through code structure.

**Why:** Comments drift from code and add maintenance burden. Good names and structure are always accurate.
```python
# ❌ Bad - comment explains unclear code
def calc(x, y, z):
    # Calculate weighted average with special handling
    if z > 0:
        return (x * 0.6 + y * 0.4) / z
    return 0

# ✅ Good - self-documenting
def calculate_weighted_average(primary_value: float, secondary_value: float, 
                               weight_divisor: float) -> float:
    PRIMARY_WEIGHT = 0.6
    SECONDARY_WEIGHT = 0.4
    
    if weight_divisor <= 0:
        return 0.0
        
    weighted_sum = (primary_value * PRIMARY_WEIGHT + 
                    secondary_value * SECONDARY_WEIGHT)
    return weighted_sum / weight_divisor
```

### Static Methods for Pure Logic

Use `@staticmethod` for stateless calculations and business logic within classes. Functions with all inputs as explicit parameters and no side effects should be static.

**Why:** Pure functions are easier to test, reuse, and understand. Clear separation between orchestration (instance methods) and computation (static methods).
```python
class OrderProcessor:
    def __init__(self, order_repo: OrderRepository, tax_calculator: TaxCalculator):
        self.order_repo = order_repo
        self.tax_calculator = tax_calculator
    
    # Instance method - orchestrates using dependencies
    def process_order(self, order_data: OrderData) -> Order:
        subtotal = self.calculate_subtotal(order_data.items)
        tax = self.tax_calculator.calculate(subtotal)
        total = subtotal + tax
        
        return self.order_repo.create(order_data, total)
    
    # Static method - pure calculation
    @staticmethod
    def calculate_subtotal(items: list[OrderItem]) -> float:
        return sum(item.price * item.quantity for item in items)
```

### Try-Finally for Resources

Always use try-finally for database connections and other resources that need cleanup. Never rely on garbage collection for resource cleanup.

**Why:** Resource leaks in production are hard to debug. Explicit cleanup ensures resources are released even when exceptions occur.
```python
# ❌ Bad - resource might leak
def get_user(user_id: int) -> User:
    conn = get_connection()
    result = conn.query(user_id)
    conn.close()  # Never called if query raises exception
    return result

# ✅ Good - guaranteed cleanup
def get_user(user_id: int) -> User:
    conn = get_connection()
    try:
        return conn.query(user_id)
    finally:
        conn.close()
```

### Never Hard-Code Secrets

All secrets and credentials must use environment variables. Never commit API keys, passwords, or tokens to version control.

**Why:** Leaked secrets in git history are permanent. Environmental configuration allows different secrets per environment.
```python
# ❌ Bad - hardcoded secret
API_KEY = "sk_live_abc123xyz"

# ✅ Good - from environment
import os
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY environment variable must be set")
```

### Clear Exception Hierarchy

Create custom exception types that map to your domain. Services raise domain exceptions, not generic exceptions.

**Why:** Domain exceptions communicate intent and allow appropriate handling at different layers.
```python
# Define clear hierarchy
class ApplicationError(Exception):
    pass

class ValidationError(ApplicationError):
    """Input doesn't meet requirements"""
    pass

class NotFoundError(ApplicationError):
    """Resource doesn't exist"""
    pass

class ConflictError(ApplicationError):
    """Business rule violated"""
    pass

# Use in code
def delete_document(doc_id: int, user_id: int) -> None:
    doc = doc_repo.get(doc_id)
    
    if not doc:
        raise NotFoundError(f"Document {doc_id} not found")
    
    if doc.owner_id != user_id:
        raise UnauthorizedError("Cannot delete document owned by another user")
    
    if doc.is_locked:
        raise ConflictError("Cannot delete locked document")
    
    doc_repo.delete(doc_id)
```

### Appropriate Logging

Log at the appropriate level for the situation. Don't log expected business errors (validation, not found) as ERROR. Log only once per error, not at every layer.

**Levels:**
- **ERROR**: Unexpected failures (database down, external API timeout, bugs)
- **WARNING**: Expected errors (validation failures, not found, unauthorized)
- **INFO**: Successful operations, important milestones
- **DEBUG**: Internal state for troubleshooting
```python
def process_order(order_id: int) -> Order:
    logger.debug(f"Processing order {order_id}")
    
    order = order_repo.get(order_id)
    if not order:
        logger.warning(f"Order not found: {order_id}")  # Expected - WARNING
        raise NotFoundError(f"Order {order_id} not found")
    
    try:
        result = payment_api.charge(order.total)
        logger.info(f"Order processed successfully: {order_id}")  # Success - INFO
        return result
        
    except PaymentAPIError as e:
        logger.error(f"Payment API failure", exc_info=True)  # Unexpected - ERROR
        raise
```

### Validate at Boundaries

Validate inputs at function entry points. Don't scatter validation throughout the codebase. Raise clear exceptions for invalid inputs.

**Why:** Centralized validation means you know exactly where to look. Early validation means bugs are caught close to their source.
```python
def create_user(name: str, email: str, age: int) -> User:
    # Validate all inputs at entry
    if not name or len(name.strip()) == 0:
        raise ValidationError("Name cannot be empty")
    
    if not email or "@" not in email:
        raise ValidationError("Invalid email format")
    
    if age < 0 or age > 150:
        raise ValidationError("Age must be between 0 and 150")
    
    # After validation, no need to check again
    return user_repo.create(name.strip(), email.lower(), age)
```

---

## Anti-Patterns

Common mistakes that violate these principles:

### ❌ God Classes
One class that does everything - violates Single Responsibility and increases cognitive load.

### ❌ Business Logic in Controllers/Views
Mixing HTTP/UI concerns with business rules - violates separation of concerns.

### ❌ Union Return Types
Returning different types based on success/failure - makes caller handle type checking.

### ❌ Magic Numbers/Strings
Unexplained constants scattered through code - should be named constants.

### ❌ Generic Exceptions
Catching `Exception` or raising `Exception("error")` - lose context about what went wrong.

### ❌ Defensive Programming Everywhere
Checking for None at every level when it should be impossible - indicates lack of trust in boundaries.

### ❌ Premature Abstraction
Creating generic solutions before understanding the pattern - wait for 3 examples.

### ❌ Multiple Sources of Truth
Same configuration/logic in multiple places - creates inconsistency.

---

## Decision Framework

When facing a design decision, ask:

1. **Cognitive Load**: Will future-me understand this in 6 months without loading context?
2. **Clarity**: Is the control flow obvious, or do I need to trace through abstractions?
3. **SSOT**: Is this the canonical place for this logic, or am I duplicating?
4. **Failure Modes**: What happens when this breaks at 2am? Will I be able to debug it?
5. **Type Safety**: Can the type system catch errors, or am I relying on runtime checks?
6. **Abstraction Level**: Am I working at the right level of detail for this task?

If the answer to any question is concerning, reconsider the approach.

---

## Summary

**Philosophy in one sentence:**  
*Build code that future-you can understand and maintain when you've forgotten all the context.*

**Implementation in one sentence:**  
*Use types, clear names, explicit control flow, and fail-early validation to make correctness obvious.*

These principles apply to every project, regardless of language, framework, or scale. Adapt the implementation to your context, but keep the philosophy consistent.
