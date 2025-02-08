import os
if __name__ == "__main__":
    from amp.standard_amp import StandardAMP
    print("#"*90)
    print("Current file:", os.path.abspath(__file__))
    print("#"*90)
    print("Class Name:", StandardAMP(learning_rate=0.01).__class__)
    print("#"*90)
else:
    from .amp.standard_amp import StandardAMP