let currentPage = 0;
const pageSize = 50;

document.getElementById('queryForm').addEventListener('submit', function(e) {
    e.preventDefault();
    currentPage = 0; // Reset to first page on new search
    queryCities();
});

document.getElementById('prevPage').addEventListener('click', function() {
    if (currentPage > 0) {
        currentPage--;
        queryCities();
    }
});

document.getElementById('nextPage').addEventListener('click', function() {
    currentPage++;
    queryCities();
});

function queryCities() {
    const cityName = document.getElementById('cityName').value;

    $.ajax({
        url: `/stat/closest_cities?city=${cityName}&page=${currentPage}&page_size=${pageSize}`,
        type: 'GET',
        success: function(data) {
            updateChart(data.closest_cities);
            document.getElementById('responseTime').innerText = `响应时间：${data.time_ms} 毫秒`;
        },
        error: function(error) {
            console.error('Error:', error);
        }
    });
}

function updateChart(citiesData) {
    const ctx = document.getElementById('distanceChart').getContext('2d');
    if (window.barChart) {
        window.barChart.destroy();
    }
    window.barChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: citiesData.map(item => item[0]),
            datasets: [{
                label: '距离',
                data: citiesData.map(item => item[1]),
                backgroundColor: 'rgba(54, 162, 235, 0.5)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}
