import sys
print("Python executable:", sys.executable)
print("Python path:")
for p in sys.path:
    print("  ", p)

try:
    import parselmouth
    print("\n✅ SUCCESS: parselmouth imported!")
    print("Version:", parselmouth.__version__)
except ImportError as e:
    print("\n❌ FAILED:", e)