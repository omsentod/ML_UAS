document.addEventListener("DOMContentLoaded", function() {
    const form = document.querySelector('form');
    const submitButton = document.querySelector('input[type="submit"]');

    // Cegah form untuk submit jika input kosong
    form.addEventListener('submit', function(event) {
        let isValid = true;
        const placeNameInput = document.querySelector('input[name="place_name"]');
        if (placeNameInput && placeNameInput.value.trim() === "") {
            isValid = false;
            alert("Nama tempat tidak boleh kosong!");
        }
        if (!isValid) {
            event.preventDefault();  // Mencegah form untuk disubmit jika input tidak valid
        }
    });

    // Show loading spinner when submitting the form
    submitButton.addEventListener("click", function() {
        submitButton.disabled = true;
        submitButton.value = "Memproses...";
    });

    // Tombol "See Tours" untuk load more recommendations
    const btn = document.querySelector('.btn');
    if (btn) {
        btn.addEventListener('click', function(event) {
            event.preventDefault();

            // Ambil nilai placeName dan offset
            let offset = parseInt(btn.getAttribute('data-offset')) || 0;
            const placeName = btn.getAttribute('data-place-name');

            // Kirim permintaan ke server
            fetch('/load_more_recommendations', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ place_name: placeName, offset: offset }),
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    console.error(data.error);
                    return;
                }

                const recommendations = data.recommendations;
                const container = document.querySelector('.tour-content');

                // Tambahkan rekomendasi baru ke dalam container
                recommendations.forEach(place => {
                    const box = document.createElement('div');
                    box.className = 'box';
                    box.innerHTML = `
                        <img src="/static/img/t2.jpg" alt="Place Image">
                        <h4>${place.City}</h4>
                        <h6>${place.Place_Name}</h6>
                        <div class="layout-prices">
                            <p><i class="fas fa-star"></i> ${place.Rating}</p>
                            <p>Price: Rp.${place.Price}</p> 
                        </div>
                    `;
                    container.appendChild(box);
                });

                // Perbarui offset untuk permintaan berikutnya
                btn.setAttribute('data-offset', offset + recommendations.length);

                // Jika tidak ada lagi rekomendasi, sembunyikan tombol
                if (recommendations.length < 5) {
                    btn.style.display = 'none';
                }
            })
            .catch(error => console.error('Error:', error));
        });
    }
});
