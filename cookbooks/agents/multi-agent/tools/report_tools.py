from typing import List, Dict
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
from .common import judgment, client

class ReportToolsMixin:
    """Mixin providing report generation and formatting tools."""
    
    @judgment.observe(span_type="tool")
    def format_report(self, title: str, content: str, sections: List[str] = None) -> str:
        """Format content into a structured report and extract chart data."""
        sections_info = f"with sections: {', '.join(sections)}" if sections else "with standard formatting"
        
        prompt = f"""
Format the following content into a professional report titled "{title}" {sections_info}.

Also extract any numerical data that could be visualized as charts. Return the response in this format:

FORMATTED REPORT:
[Your formatted report here]

CHART DATA:
[JSON dictionary with chart data, where each key is a chart name and value contains the data]

Content: {content}

Example chart data format:
{{"Revenue by Quarter": {{"values": [100, 150, 200], "labels": ["Q1", "Q2", "Q3"]}}, "Growth Rate": [10, 15, 20]}}
"""
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You are a professional report writer who also extracts data for visualization."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    
    @judgment.observe(span_type="tool")
    def create_executive_summary(self, findings: List[str], recommendations: List[str] = None) -> str:
        """Create an executive summary from findings and recommendations."""
        findings_text = "\n".join([f"- {finding}" for finding in findings])
        recommendations_text = "\n".join([f"- {rec}" for rec in recommendations]) if recommendations else "None provided"
        
        prompt = f"""
Create a professional executive summary based on these findings and recommendations.

Key Findings:
{findings_text}

Recommendations:
{recommendations_text}

Format as a complete executive summary with:
1. Overview
2. Key Findings section
3. Recommendations section  
4. Conclusion

Executive Summary:
"""
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You are a professional business analyst creating executive summaries."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    
    @judgment.observe(span_type="tool")
    def generate_charts_description(self, data: Dict) -> str:
        """Generate actual charts as PNG files based on data dictionary."""
        try:
            # Handle case where data might be a string (from format_report output)
            if isinstance(data, str):
                # Try to extract JSON from the format_report output
                if "CHART DATA:" in data:
                    chart_data_section = data.split("CHART DATA:")[1].strip()
                    # Find the JSON part
                    lines = chart_data_section.split('\n')
                    json_line = None
                    for line in lines:
                        line = line.strip()
                        if line.startswith('{') and line.endswith('}'):
                            json_line = line
                            break
                    
                    if json_line:
                        try:
                            data = json.loads(json_line)
                        except json.JSONDecodeError:
                            return "Error: Could not parse chart data from report output"
                    else:
                        return "No chart data found in the provided input"
                else:
                    return "No chart data section found in the provided input"
            
            if not isinstance(data, dict) or not data:
                return "No valid chart data provided"
            
            # Create output directory if it doesn't exist
            output_dir = "reports"
            os.makedirs(output_dir, exist_ok=True)
            
            generated_files = []
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            for chart_name, chart_data in data.items():
                filename = f"{output_dir}/{chart_name.replace(' ', '_')}_{timestamp}.png"
                
                # Create a simple chart based on data type
                plt.figure(figsize=(10, 6))
                
                if isinstance(chart_data, dict) and 'values' in chart_data:
                    # Bar chart for values with labels
                    values = chart_data['values']
                    labels = chart_data.get('labels', [f"Item {i+1}" for i in range(len(values))])
                    plt.bar(labels, values)
                    plt.title(f"{chart_name}")
                    plt.ylabel("Values")
                    
                elif isinstance(chart_data, dict):
                    # Bar chart for dictionary data
                    labels = list(chart_data.keys())
                    values = [float(v) if isinstance(v, (int, float)) else len(str(v)) for v in chart_data.values()]
                    plt.bar(labels, values)
                    plt.title(f"{chart_name}")
                    plt.xticks(rotation=45)
                    plt.ylabel("Values")
                    
                elif isinstance(chart_data, list):
                    # Line chart for list data
                    numeric_data = [float(x) if isinstance(x, (int, float)) else len(str(x)) for x in chart_data]
                    plt.plot(numeric_data, marker='o')
                    plt.title(f"{chart_name}")
                    plt.xlabel("Index")
                    plt.ylabel("Values")
                    
                else:
                    # Simple text chart
                    plt.text(0.5, 0.5, f"{chart_name}: {chart_data}", 
                            horizontalalignment='center', verticalalignment='center',
                            transform=plt.gca().transAxes, fontsize=14)
                    plt.title(f"{chart_name}")
                    plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close()
                
                generated_files.append(filename)
            
            if generated_files:
                files_list = "\n".join([f"- {f}" for f in generated_files])
                return f"Charts generated successfully:\n{files_list}\n\nChart data used:\n{json.dumps(data, indent=2)}"
            else:
                return "No charts generated - no valid data provided"
                
        except Exception as e:
            return f"Error generating charts: {str(e)}" 