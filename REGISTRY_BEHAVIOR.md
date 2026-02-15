# Registry Behavior Guide 🎯

## Before vs After

### 🔴 BEFORE (Problematic)

```
User doesn't have sqlite3 installed
                 ↓
         Import registry
                 ↓
    ❌ CRASH: ModuleNotFoundError: No module named 'sqlite3'
                 ↓
         Application fails
```

**Result**: Users without sqlite3 can't use OctoLearn at all 😞

---

### 🟢 AFTER (Fixed)

```
User doesn't have sqlite3 installed
                 ↓
         Import registry
                 ↓
   ✅ sqlite3 check: SKIPPED (not available)
                 ↓
   ⚠️ WARNING: "sqlite3 not available. Falling back to JSON..."
                 ↓
   ✅ Use JSON storage (built-in, no deps)
                 ↓
    Everything works perfectly! 🎉
```

**Result**: Works for everyone, regardless of sqlite3 ✨

---

## Storage Options Decision Tree

```
Registry needed?
    │
    ├─→ Yes, want max portability?
    │   └─→ JSON ✅ (DEFAULT, NO DEPS)
    │
    ├─→ Yes, want human-readable in spreadsheet?
    │   └─→ CSV ✅ (NEW, NO DEPS)
    │
    └─→ Yes, want advanced relational features?
        └─→ SQLite (OPTIONAL, INSTALLED?)
            ├─→ Yes → SQLite ✅
            └─→ No → Falls back to JSON ✅
```

---

## How It Works

### Import Phase
```python
# SAFE - No crash even if sqlite3 missing
try:
    import sqlite3
    SQLITE_AVAILABLE = True  # ✅
except ImportError:
    SQLITE_AVAILABLE = False  # ✅ Still works!
```

### Initialization Phase
```python
registry = ModelRegistry(storage_type='sqlite')

# System checks: Is sqlite3 available?
if SQLITE_AVAILABLE:
    # ✅ User gets SQLite
    return 'sqlite'
else:
    # ✅ Automatic fallback to JSON
    logger.warning("Falling back to JSON...")
    return 'json'
```

---

## Code Examples

### Scenario 1: Default Setup (Recommended)
```python
from octolearn import AutoML

automl = AutoML(
    train_models=True,      # Train models
    use_registry=True       # Save to registry
)

automl.fit(X, y)

# 🎯 What happens:
# 1. Trainer completes
# 2. Registry initializes with JSON (default)
# 3. Models saved to: .octolearn/model_registry.json
# ✅ No dependencies needed!
```

**Output**:
```json
// .octolearn/model_registry.json
{
  "RandomForest_v1": {
    "name": "RandomForest",
    "version": 1,
    "task_type": "classification",
    "metrics": {...},
    "timestamp": "2026-02-15T10:30:00",
    "model_path": "RandomForest_v1.pkl"
  },
  "XGBoost_v1": {...}
}
```

### Scenario 2: User Prefers CSV
```python
from octolearn.models import ModelRegistry

# CSV format - auditable in Excel
registry = ModelRegistry(storage_type='csv')
registry.register_model('MyModel', model, 'classification')

# ✅ Creates human-readable CSV file
```

**Output**:
```csv
model_id,name,version,task_type,timestamp,model_path
RandomForest_v1,RandomForest,1,classification,2026-02-15T10:30:00,RandomForest_v1.pkl
XGBoost_v1,XGBoost,1,classification,2026-02-15T10:35:00,XGBoost_v1.pkl
```

### Scenario 3: ᴘᴏʷᴇʀ ᴜsᴇʀ with SQLite
```python
from octolearn.models import ModelRegistry

# User has sqlite3 installed
registry = ModelRegistry(storage_type='sqlite')

# ✅ Uses SQLite for advanced features
# ✅ Works perfectly with dependencies
```

### Scenario 4: Requested SQLite, But Missing
```python
# User tries SQLite but it's not installed
registry = ModelRegistry(storage_type='sqlite')

# 🎯 What happens:
# ⚠️ Warning: "sqlite3 not available. Falling back to JSON..."
# ✅ Automatically switches to JSON
# 0️⃣ No crash!
```

---

## Storage Comparison

| Feature | JSON | CSV | SQLite |
|---------|------|-----|--------|
| **Dependencies** | None ✅ | None ✅ | Optional |
| **Default** | Yes ✅ | No | No |
| **Human-Readable** | Yes ✅ | Yes ✅ | No |
| **Excel Import** | Need parser | Direct ✅ | No |
| **Version Control** | Perfect ✅ | Perfect ✅ | Not ideal |
| **Relational Queries** | No | No | Yes ✅ |
| **Backup** | Simple ✅ | Simple ✅ | Complex |
| **Portability** | Excellent ✅ | Excellent ✅ | Conditional |

---

## Migration Guide (For Existing Users)

If you're currently using SQLite:

### Option 1: Continue with SQLite (if installed)
```python
# Keep using SQLite - it still works!
registry = ModelRegistry(storage_type='sqlite')
# ✅ Nothing changes for you
```

### Option 2: Switch to JSON (Recommended)
```python
# Switch to JSON for portability
registry = ModelRegistry(storage_type='json')
# ✅ Better for version control and backups
```

### Option 3: Use CSV
```python
# Switch to CSV for auditing
registry = ModelRegistry(storage_type='csv')
# ✅ Open in Excel, track changes easily
```

---

## Error Handling

### ✅ Safe Operations

```python
# All of these are safe now:

registry = ModelRegistry()              # ✅ Uses JSON (default)
registry = ModelRegistry(storage_type='sqlite')  # ✅ Falls back to JSON if needed
registry = ModelRegistry(storage_type='json')    # ✅ Works
registry = ModelRegistry(storage_type='csv')     # ✅ Works
registry = ModelRegistry(storage_type='unknown') # ✅ Falls back to JSON with warning
```

### 🔴 Even if sqlite3 Missing

```python
# Before: ❌ CRASH
import sqlite3  # ModuleNotFoundError

# After: ✅ WORKS
try:
    import sqlite3
except ImportError:
    pass  # Still works!
```

---

## Summary Table

| User Situation | Default Behavior | Result |
|---|---|---|
| **No sqlite3, uses default** | JSON | ✅ Works perfectly |
| **No sqlite3, requests SQLite** | Fallback to JSON | ✅ Works with warning |
| **Has sqlite3, uses default** | JSON | ✅ Works perfectly |
| **Has sqlite3, requests SQLite** | SQLite | ✅ Works with full features |
| **Requests CSV** | CSV | ✅ Works, human-readable |

---

## 🎉 Your OctoLearn is Now Production-Ready

✅ **Zero-Dependency Registry** (JSON)
✅ **Automatic Fallback System**
✅ **Multiple Storage Options**
✅ **No Breaking Changes**
✅ **Perfect Portability**

---

Get started:
```python
from octolearn import AutoML

automl = AutoML(train_models=True, use_registry=True)
automl.fit(X, y)
# 🎯 Models automatically saved and versioned!
```

**That's it!** No configuration needed. Everything just works. 🚀
