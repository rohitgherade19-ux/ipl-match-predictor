document.addEventListener("DOMContentLoaded", function() {
    const team1Select = document.getElementById("team1");
    const team2Select = document.getElementById("team2");
    const tossWinnerSelect = document.getElementById("toss_winner");
    const teamAlert = document.getElementById("teamAlert");
    const submitBtn = document.getElementById("submitBtn");
    const form = document.getElementById("predictionForm");
    const spinner = document.getElementById("submitSpinner");

    function updateTossWinnerOptions() {
        if (!team1Select || !team2Select || !tossWinnerSelect) return;

        const team1 = team1Select.value;
        const team2 = team2Select.value;

        // Clear options
        tossWinnerSelect.innerHTML = '<option value="" selected disabled>Select Toss Winner</option>';

        if (team1 && team2) {
            tossWinnerSelect.disabled = false;
            tossWinnerSelect.nextElementSibling.classList.add("d-none"); // hide "please select teams" msg

            // Add Team 1
            const opt1 = document.createElement("option");
            opt1.value = team1;
            opt1.textContent = team1;
            tossWinnerSelect.appendChild(opt1);

            // Add Team 2
            if (team1 !== team2) {
                const opt2 = document.createElement("option");
                opt2.value = team2;
                opt2.textContent = team2;
                tossWinnerSelect.appendChild(opt2);
            }
        } else {
            tossWinnerSelect.disabled = true;
            tossWinnerSelect.nextElementSibling.classList.remove("d-none");
        }

        validateTeams();
    }

    function validateTeams() {
        if (!team1Select || !team2Select) return;
        const team1 = team1Select.value;
        const team2 = team2Select.value;

        if (team1 && team2 && team1 === team2) {
            teamAlert.classList.remove("d-none");
            submitBtn.disabled = true;
        } else {
            teamAlert.classList.add("d-none");
            submitBtn.disabled = false;
        }
    }

    if (team1Select) team1Select.addEventListener("change", updateTossWinnerOptions);
    if (team2Select) team2Select.addEventListener("change", updateTossWinnerOptions);

    if (form) {
        form.addEventListener("submit", function(e) {
            if (team1Select.value === team2Select.value) {
                e.preventDefault();
                validateTeams();
                return;
            }
            // Show spinner on valid submit
            submitBtn.disabled = true;
            if (spinner) spinner.classList.remove("d-none");
            form.submit();
        });
    }

    // Smooth scroll for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            document.querySelector(this.getAttribute('href')).scrollIntoView({
                behavior: 'smooth'
            });
        });
    });
});
