async function searchICD() {
    const desc = document.getElementById("description").value;
    if (!desc) return alert("Please enter a description");

    const response = await fetch(`/search_icd?description=${encodeURIComponent(desc)}`);
    const data = await response.json();

    const resultsDiv = document.getElementById("results");
    resultsDiv.innerHTML = "";

    // If no data
    if (!data["ICD Code"] || data["ICD Code"].length === 0) {
        resultsDiv.innerHTML = `<p class="placeholder">No matching ICD codes found.</p>`;
        return;
    }

    // Build Table
    let table = `
        <table class="results-table">
            <thead>
                <tr>
                    <th>#</th>
                    <th>ICD Code</th>
                    <th>Similarity</th>
                </tr>
            </thead>
            <tbody>
    `;

    for (let i = 0; i < data["ICD Code"].length; i++) {
        table += `
            <tr>
                <td>${i + 1}</td>
                <td>${data["ICD Code"][i]}</td>
                <td>${parseFloat(data["Similarity"][i]).toFixed(3)}</td>
            </tr>
        `;
    }

    table += "</tbody></table>";
    resultsDiv.innerHTML = table;
}
