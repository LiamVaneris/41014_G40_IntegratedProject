
window.HELP_IMPROVE_VIDEOJS = false;
 
$(document).ready(function () {
  // Navbar burger toggle for small screens
  $(".navbar-burger").click(function () {
    $(".navbar-burger").toggleClass("is-active");
    $(".navbar-menu").toggleClass("is-active");
  });
 
  // Team members carousel
  // Looping is switched off because infinite mode clones slides and that is
  // what caused the cloneNode error. It also only attaches once per element.
  var carouselEl = document.querySelector("#results-carousel");
 
  if (carouselEl && !carouselEl.dataset.carouselReady) {
    carouselEl.dataset.carouselReady = "true";
 
    var slideCount = carouselEl.querySelectorAll(".item").length;
 
    try {
      bulmaCarousel.attach("#results-carousel", {
        slidesToScroll: 1,
        slidesToShow: Math.min(3, slideCount),
        loop: false,
        infinite: false,
        autoplay: false,
        pagination: true,
        navigation: true
      });
    } catch (err) {
      console.error("Carousel failed to start", err);
    }
  }
 
  // Range sliders (only runs if any exist on the page)
  if (typeof bulmaSlider !== "undefined") {
    bulmaSlider.attach();
  }
});