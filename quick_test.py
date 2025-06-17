#!/usr/bin/env python3
"""
Simple API test using requests
"""

import requests
import json

def test_api():
    base_url = "http://localhost:8000"
    
    print("Testing UNR-Explainer API")
    print("=" * 50)
    
    # Test 1: Health check
    try:
        print("1. Testing health check...")
        response = requests.get(f"{base_url}/api/v1/health")
        if response.status_code == 200:
            health = response.json()
            print(f"✅ Health check passed: {health}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return
    
    # Test 2: Available models
    try:
        print("\n2. Testing available models...")
        response = requests.get(f"{base_url}/api/v1/models")
        if response.status_code == 200:
            models = response.json()
            print(f"✅ Models available: {models}")
        else:
            print(f"❌ Models check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Models check error: {e}")
    
    # Test 3: Single node explanation (Cora)
    try:
        print("\n3. Testing single node explanation (Cora)...")
        params = {
            "dataset": "Cora",
            "model": "graphsage", 
            "task": "node",
            "node_id": 0,
            "timeout": 30
        }
        response = requests.post(f"{base_url}/api/v1/explain/single", params=params)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Cora explanation success: importance={result.get('importance', 'N/A')}")
        else:
            print(f"❌ Cora explanation failed: {response.status_code}")
            try:
                error_detail = response.json()
                print(f"   Error details: {error_detail}")
            except:
                print(f"   Error text: {response.text}")
    except Exception as e:
        print(f"❌ Cora explanation error: {e}")
    
    # Test 4: Single node explanation (DBLP)
    try:
        print("\n4. Testing single node explanation (DBLP)...")
        params = {
            "dataset": "DBLP",
            "model": "magnn",
            "task": "node", 
            "node_id": 100,
            "timeout": 30
        }
        response = requests.post(f"{base_url}/api/v1/explain/single", params=params)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ DBLP explanation success: importance={result.get('importance', 'N/A')}")
        else:
            print(f"❌ DBLP explanation failed: {response.status_code}")
            try:
                error_detail = response.json()
                print(f"   Error details: {error_detail}")
            except:
                print(f"   Error text: {response.text}")
    except Exception as e:
        print(f"❌ DBLP explanation error: {e}")

if __name__ == "__main__":
    test_api() 