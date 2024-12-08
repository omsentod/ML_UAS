// File: static/js/script.js

// Script untuk menangani form submission
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
});
