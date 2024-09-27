var user = JSON.parse(localStorage.getItem('user')); // mengambil data pengguna dari local storage
    var interval=document.getElementById('interval')
    var navbarUser = document.getElementById('navbarUser');
    var navbarRegister = document.getElementById('navbarRegister');
    var logoutButton = document.getElementById('logoutButton');

    if (user !== null) { // jika pengguna sudah login
        navbarUser.textContent = 'Hi! '+ user.name; // menampilkan nama pengguna
        navbarUser.style.display = 'inline'; // menampilkan elemen nama pengguna
        navbarRegister.style.display='none';
        interval.style.display='inline';
    }
    navbarUser.addEventListener('click', function() {
      logoutButton.style.display = 'inline'; // menampilkan tombol logout
  });
    logoutButton.addEventListener('click', function() {
        localStorage.removeItem('user'); // menghapus data pengguna dari local storage
        navbarUser.style.display = 'none'; // menyembunyikan elemen nama pengguna
        logoutButton.style.display = 'none'; // menyembunyikan tombol logout
        navbarRegister.style.display='inline';
        interval.style.display='none'
    });
    window.addEventListener('click', function(event) {
      if (event.target !== logoutButton && event.target !== navbarUser) {
          logoutButton.style.display = 'none'; // menyembunyikan tombol logout jika pengguna mengklik di luar tombol logout
      }
  });