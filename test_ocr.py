#!/usr/bin/env python3
import sys
sys.path.insert(0, '/app')

from web_app.app import get_model

model = get_model()
print("Model loaded successfully")

with open('/tmp/test_image.png', 'rb') as f:
    text = model.extract_text(f.read())
    print('Extracted text:', text)
