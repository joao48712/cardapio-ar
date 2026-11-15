const err = new URLSearchParams(location.search).get("error");
if (err) {
  document.getElementById("msg").textContent =
    "O motor Python não iniciou. Rode “npm run setup” na pasta ia-paginada e tente de novo.";
  const pre = document.getElementById("err");
  pre.hidden = false;
  pre.textContent = err;
  document.querySelector(".book").hidden = true;
}
