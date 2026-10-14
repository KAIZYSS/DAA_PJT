import requests
import re
import urllib3
urllib3.disable_warnings()

try:
    r = requests.get('https://www.tngis.tn.gov.in/apps/accident-dashboard/', verify=False, timeout=15)
    print('Status:', r.status_code)
    print('Length:', len(r.text))
    scripts = re.findall(r'<script[^>]*src=["\']([^"\']+)["\']', r.text)
    print('Scripts found:', len(scripts))
    for s in scripts:
        print('  Script:', s)
except Exception as e:
    print('Error:', e)

for s in ['assets/scripts/main.js', 'assets/scripts/dependent.js']:
    url = 'https://www.tngis.tn.gov.in/apps/accident-dashboard/' + s
    try:
        r = requests.get(url, verify=False, timeout=10)
        print(s, 'Status:', r.status_code, 'Length:', len(r.text))
        matches = re.findall(r'[\'"](/[^\'"\s]+|https?://[^\'"\s]+)[\'"]', r.text)
        endpoints = [m for m in matches if any(x in m.lower() for x in ['api', 'service', 'get', 'post', 'wms', 'wfs', 'data', 'json', 'php', 'ajax'])]
        print('  Endpoints in', s, ':', set(endpoints))
        ajax_calls = re.findall(r'url\s*:\s*[\'"]([^\'"]+)[\'"]', r.text)
        print('  Ajax URLs:', set(ajax_calls))
    except Exception as e:
        print('Error fetching', s, e)
