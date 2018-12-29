import sys
import importlib

# sys.path.insert(0, '')

model = importlib.import_module('model')
model.build_model()
