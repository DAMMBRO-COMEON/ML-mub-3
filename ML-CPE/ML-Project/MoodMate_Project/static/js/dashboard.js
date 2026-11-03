async function loadRealDashboard() {
    const response = await fetch('/api/dashboard_data');
    const data = await response.json();

    const ctxDaily = document.getElementById('dailyMoodChart').getContext('2d');
    new Chart(ctxDaily, {
        type: 'line',
        data: {
            labels: data.daily_labels.length > 0 ? data.daily_labels : ['ยังไม่มีข้อมูล'],
            datasets: [{
                label: 'คะแนนความรู้สึกเฉลี่ย (1-10)',
                data: data.daily_data.length > 0 ? data.daily_data : [0],
                borderColor: 'rgb(75, 192, 192)',
                tension: 0.2
            }]
        },
        options: { scales: { y: { min: 0, max: 10 } } }
    });

    const ctxMonthly = document.getElementById('monthlyMoodChart').getContext('2d');
    new Chart(ctxMonthly, {
        type: 'doughnut',
        data: {
            labels: data.monthly_labels,
            datasets: [{
                data: data.monthly_data,
                backgroundColor: ['#28a745', '#007bff', '#6c757d', '#ffc107', '#dc3545']
            }]
        }
    });
}

loadRealDashboard();