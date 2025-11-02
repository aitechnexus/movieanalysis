#!/bin/bash

echo "🧪 Starting MovieLens Analysis Testing..."
echo "========================================"

# Test API endpoints
echo ""
echo "📡 Testing API endpoints..."
endpoints=("status" "health" "overview" "user-stats" "comprehensive-statistics" "advanced-heatmaps" "percentage-analysis")

for endpoint in "${endpoints[@]}"; do
    echo -n "Testing /api/$endpoint... "
    response=$(curl -s -w "%{http_code}" http://localhost:8001/api/$endpoint -o /dev/null)
    if [[ "$response" == "200" ]]; then
        echo "✅ PASS"
    else
        echo "❌ FAIL (HTTP $response)"
    fi
done

# Test frontend pages
echo ""
echo "🌐 Testing frontend pages..."
declare -a pages=("" "trends.html" "dataset.html" "documentation.html")
declare -a page_names=("Main Dashboard" "Trends Analysis" "Dataset Management" "Documentation")

for i in "${!pages[@]}"; do
    page_url="${pages[$i]}"
    page_name="${page_names[$i]}"
    echo -n "Testing $page_name (/$page_url)... "
    response=$(curl -s -w "%{http_code}" http://localhost:8000/$page_url -o /dev/null)
    if [[ "$response" == "200" ]]; then
        echo "✅ PASS"
    else
        echo "❌ FAIL (HTTP $response)"
    fi
done

# Test Docker services
echo ""
echo "🐳 Testing Docker services..."
echo -n "Checking Docker containers... "
if docker compose ps --format "table {{.Service}}\t{{.Status}}" | grep -q "Up"; then
    echo "✅ PASS"
    docker compose ps --format "table {{.Service}}\t{{.Status}}"
else
    echo "❌ FAIL - Some containers are not running"
    docker compose ps
fi

# Test performance
echo ""
echo "⚡ Testing performance..."
echo -n "Testing cached endpoint response time... "
start_time=$(date +%s%N)
curl -s http://localhost:8001/api/advanced-heatmaps > /dev/null 2>&1
end_time=$(date +%s%N)
duration=$(( (end_time - start_time) / 1000000 ))

if [[ $duration -lt 100 ]]; then
    echo "✅ PASS ($duration ms)"
else
    echo "⚠️  SLOW ($duration ms - expected <100ms)"
fi

# Test cache files
echo ""
echo "💾 Testing cache system..."
echo -n "Checking cache directories... "
if [[ -d "data/cache" && -d "data/cache/visualizations" ]]; then
    echo "✅ PASS"
else
    echo "❌ FAIL - Cache directories missing"
fi

echo -n "Checking cache files... "
if [[ -f "data/cache/analysis_cache.json" ]]; then
    cache_size=$(stat -f%z "data/cache/analysis_cache.json" 2>/dev/null || stat -c%s "data/cache/analysis_cache.json" 2>/dev/null)
    if [[ $cache_size -gt 0 ]]; then
        echo "✅ PASS (${cache_size} bytes)"
    else
        echo "⚠️  EMPTY cache file"
    fi
else
    echo "❌ FAIL - Cache file missing"
fi

# Summary
echo ""
echo "🎉 Testing complete!"
echo "========================================"
echo ""
echo "📋 Quick Health Check:"
echo "- Frontend: http://localhost:8000/"
echo "- Backend API: http://localhost:8001/api/status"
echo "- Documentation: http://localhost:8000/documentation.html"
echo ""
echo "💡 If any tests failed, check the troubleshooting section in the documentation."