import json
import os
import time
from openai import OpenAI
from dotenv import load_dotenv

# Load API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def process_item(item):
    """Classify, summarise and suggest action for a single item."""
    
    prompt = f"""You are an AI operations assistant. Analyse this item and respond in JSON format.

Item type: {item['type']}
Content: {item['content']}

Respond with exactly this JSON structure:
{{
    "category": "one of: Training, Technical Issue, Strategy, Integration, Feedback",
    "priority": "one of: High, Medium, Low",
    "summary": "one sentence summary",
    "action": "one specific recommended next action"
}}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    
    result = json.loads(response.choices[0].message.content)
    return result

def generate_report(items, results):
    """Generate a Markdown report from the results."""
    
    report = "# AI Operations Triage Report\n\n"
    report += f"**Total items processed:** {len(items)}\n\n"
    report += "---\n\n"
    
    for item, result in zip(items, results):
        report += f"## Item {item['id']} — {result['category']}\n"
        report += f"**Type:** {item['type']}  \n"
        report += f"**Priority:** {result['priority']}  \n"
        report += f"**Summary:** {result['summary']}  \n"
        report += f"**Recommended Action:** {result['action']}  \n\n"
        report += "---\n\n"
    
    return report

def main():
    print("🤖 AI Batch Processor\n")
    
    # Load items
    with open("inputs/items.json", "r") as f:
        items = json.load(f)
    
    print(f"Processing {len(items)} items...\n")
    
    results = []
    for item in items:
        print(f"  Processing item {item['id']}...")
        result = process_item(item)
        results.append(result)
        time.sleep(1)
    
    # Save JSON results
    with open("outputs/results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Generate and save Markdown report
    report = generate_report(items, results)
    with open("outputs/report.md", "w") as f:
        f.write(report)
    
    print(f"\n✅ Done!")
    print(f"   outputs/results.json — structured data")
    print(f"   outputs/report.md — readable report")

if __name__ == "__main__":
    main()
    