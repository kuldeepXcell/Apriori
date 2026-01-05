"""
Streamlit component that captures a PNG snapshot of a rendered Vega-Lite chart.
"""

from __future__ import annotations

import base64
from typing import Any

import streamlit.components.v1 as components


def capture_chart_png(vega_lite_spec: dict[str, Any], trigger: int, height: int = 600) -> bytes | None:
    """
    Render a Vega-Lite spec in the browser and capture it as PNG bytes.
    
    Args:
        vega_lite_spec: Complete Vega-Lite specification
        trigger: Unique trigger number to force re-render
        height: Component height in pixels
        
    Returns:
        PNG bytes if successful, None on first render (component needs to post back)
    """
    html_template = """
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
    <script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
    <script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
</head>
<body style="margin:0;padding:0;">
    <!-- trigger: TRIGGER_ID -->
    <div id="vis"></div>
    <script type="text/javascript">
        const spec = SPEC_PLACEHOLDER;
        
        vegaEmbed('#vis', spec, {actions: false})
            .then(result => {
                // Wait for full render
                setTimeout(() => {
                    result.view.toImageURL('png', 2)
                        .then(url => {
                            // Extract base64 from data URL
                            const base64Data = url.split(',')[1];
                            // Send back to Streamlit
                            window.parent.postMessage({
                                type: 'streamlit:setComponentValue',
                                value: base64Data
                            }, '*');
                        })
                        .catch(err => {
                            console.error('PNG export failed:', err);
                            window.parent.postMessage({
                                type: 'streamlit:setComponentValue',
                                value: 'ERROR'
                            }, '*');
                        });
                }, 300);
            })
            .catch(err => {
                console.error('vegaEmbed failed:', err);
                window.parent.postMessage({
                    type: 'streamlit:setComponentValue',
                    value: 'ERROR'
                }, '*');
            });
    </script>
</body>
</html>
"""
    
    import json
    html_content = html_template.replace("SPEC_PLACEHOLDER", json.dumps(vega_lite_spec))
    html_content = html_content.replace("TRIGGER_ID", str(trigger))
    
    # Render component and wait for base64 response
    base64_data = components.html(html_content, height=height, scrolling=False)
    
    if base64_data == "ERROR":
        return None
    if base64_data:
        try:
            return base64.b64decode(base64_data)
        except Exception:
            return None
    return None

