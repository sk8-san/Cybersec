particlesJS("particles-js", {
  "particles": {
    "number": {
      "value": 155,
      "density": {
        "enable": true,
        "value_area": 789.1476416322727
      }
    },
    "color": {
      "value": "#000000"
    },
    "shape": {
      "type": "circle",
      "stroke": {
        "width": 0,
        "color": "#000000"
      },
      "polygon": {
        "nb_sides": 5
      },
      "image": {
        "src": "img/github.svg",
        "width": 100,
        "height": 100
      }
    },
    "opacity": {
      "value": 0.48927153781200905,
      "random": false,
      "anim": {
        "enable": true,
        "speed": 1,
        "opacity_min": 0,
        "sync": false
      }
    },
    "size": {
      "value": 2,
      "random": true,
      "anim": {
        "enable": true,
        "speed": 2,
        "size_min": 0,
        "sync": false
      }
    },
    "line_linked": {
      "enable": true,
      "distance": 150,
      "color": "#000000",
      "opacity": 0.4,
      "width": 1
    },
    "move": {
      "enable": true,
      "speed": 0.2,
      "direction": "none",
      "random": true,
      "straight": false,
      "out_mode": "out",
      "bounce": false,
      "attract": {
        "enable": false,
        "rotateX": 600,
        "rotateY": 1200
      }
    }
  },
  "interactivity": {
    "detect_on": "canvas",
    "events": {
      "onhover": {
        "enable": true,
        "mode": "bubble"
      },
      "onclick": {
        "enable": true,
        "mode": "push"
      },
      "resize": true
    },
    "modes": {
      "grab": {
        "distance": 400,
        "line_linked": {
          "opacity": 1
        }
      },
      "bubble": {
        "distance": 83.91608391608392,
        "size": 1,
        "duration": 3,
        "opacity": 1,
        "speed": 3
      },
      "repulse": {
        "distance": 200,
        "duration": 0.4
      },
      "push": {
        "particles_nb": 4
      },
      "remove": {
        "particles_nb": 2
      }
    }
  },
  "retina_detect": true
});

$(function() {
  var data = [
    {
      action: 'type',
      strings: ["What is this Website?"],
      output: '<span class="gray">This website checks an executable file is a malware or not.</><br><br>',
      postDelay: 1000
    },
  { 
    action: 'type',
    strings: ["Why should I use this?"],
    output: '<span class="gray">This is a way to check if your file to safe, without you having to open it. This uses stateless inspection method and only checks header of the executable.</><br>&nbsp;',
    postDelay: 1000
  },
  { 
    action: 'type',
    strings: ["You may now proceed to upload your file......" ],
    postDelay: 1000
  }
  
];
  runScripts(data, 0);
});

function runScripts(data, pos) {
    var prompt = $('.prompt'),
        script = data[pos];
    if(script.clear === true) {
      $('.history').html(''); 
    }
    switch(script.action) {
        case 'type':
          // cleanup for next execution
          prompt.removeData();
          $('.typed-cursor').text('');
          prompt.typed({
            strings: script.strings,
            typeSpeed: 30,
            callback: function() {
              var history = $('.history').html();
              history = history ? [history] : [];
              history.push('$ ' + prompt.text());
              if(script.output) {
                history.push(script.output);
                prompt.html('');
                $('.history').html(history.join('<br>'));
              }
              // scroll to bottom of screen
              $('section.terminal').scrollTop($('section.terminal').height());
              // Run next script
              pos++;
              if(pos < data.length) {
                setTimeout(function() {
                  runScripts(data, pos);
                }, script.postDelay || 1000);
              }
            }
          });
          break;
        case 'view':

          break;
    }
}
let fileInput = document.getElementById("file-upload-input");
let fileSelect = document.getElementsByClassName("file-upload-select")[0];
fileSelect.onclick = function() {
	fileInput.click();
}
fileInput.onchange = function() {
	let filename = fileInput.files[0].name;
	let selectName = document.getElementsByClassName("file-select-name")[0];
	selectName.innerText = filename;
}
const bottom = document.querySelector(".bottom");
const overlay = document.querySelector(".overlay");
const count = 110;
const size = 50;
for (let i = 0; i <= count; i += 1) {
  const dot = document.createElement("div");
  dot.classList.add("dot");
  bottom.appendChild(dot);
}
const dots = Array.from(document.querySelectorAll(".dot"));

const updateText = (text) => {
  Array.from(document.querySelectorAll(".text")).forEach(
    (e) => (e.innerHTML = text)
  );
};

const reset = () => {
  dots.forEach((dot, i) => {
    const x = (i / count) * (190 + size) - size / 2;
    const y = Math.random(1) * 52 - size / 2;
    dot.style.width = `${size}px`;
    dot.style.height = `${size}px`;
    dot.style.left = `${x}px`;
    dot.style.top = `${y}px`;
    dot.style.opacity = 1;
    dot.style.transform = "scale(1)";
  });
};
reset();

overlay.addEventListener("click", () => {
  anime({
    easing: "linear",
    targets: document.querySelectorAll(".dot"),
    opacity: [{ value: 0, duration: 600, delay: anime.stagger(10) }],
    translateX: {
      value: function () {
        return anime.random(-30, 30);
      },
      duration: 400,
      delay: anime.stagger(10)
    },
    translateY: {
      value: function () {
        return anime.random(-30, 30);
      },
      duration: 400,
      delay: anime.stagger(10)
    },
    scale: {
      value: function () {
        return 0;
      },
      duration: 400,
      delay: anime.stagger(10)
    }
  });
  anime({
    easing: "linear",
    delay: 4000,
    complete: () => {
      updateText("Submitted");
      setTimeout(() => {
        updateText("Submit");
        reset();
      }, 3000);
    }
  });
});