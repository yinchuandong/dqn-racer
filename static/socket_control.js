(function(){
  var socket = io('http://127.0.0.1:5000');
  var jStatus = Dom.get('server-status');

  socket.on('connect', function(){
    jStatus.innerHTML = '<p>connect</p>';
  });
  
  socket.on('disconnect', function(){
    jStatus.innerHTML += '<p>disconnect</p>';
  });

  socket.on('init', function(action){
    timeIntervalID = setInterval(function(){
      var json = capture();
      socket.emit('message', json);
    }, 1000/30);
  });

  socket.on('message', function(action){

  });

  var timeIntervalID = null;
  // dynamically create a smaller canvas for preview
  var smallCanvas = document.createElement('canvas');
  var smallCtx = smallCanvas.getContext('2d');
  var smallImage = new Image();
  var smallWidth = 84;
  var smallHeight = 84;
  smallCanvas.width = smallWidth;
  smallCanvas.height = smallHeight;
  smallCanvas.zIndex = '100';
  Dom.get('preview').appendChild(smallCanvas);

  // scale the snapshot of main canvas to a smaller one
  function scaleImage(){
    smallImage.src = canvas.toDataURL();
    smallCtx.drawImage(smallImage, 0, 0, smallWidth, smallHeight);
    var prefix = 'data:image/png;base64,';
    return smallCanvas.toDataURL().substr(prefix.length);
  }

  function capture() {
    var data = scaleImage();
    // send status to sever for calculating reward
    var json = {
      'img': data,
      'status': {
        collision: COLLISION_OCCURED,
        terminal: TERMINAL,
        start_frame: START_FRAME
        action: [keyLeft, keyRight, keyFaster, keySlower]
      },
      'telemetry': {
        playerX: playerX,
        speed: speed,
        maxSpeed: maxSpeed,
      } 
    }
    START_FRAME = false;
    return json;
  };

  setTimeout(function(){
    Game.run(gameParams);
    var json = capture();
    socket.emit('init', json);
  }, 1000);

//=========================================================================
// THE GAME LOOP
//=========================================================================
  var gameParams = {
    canvas: canvas, render: render, update: update, stats: stats, step: step,
    images: ["background", "sprites"],
    keys: [
      { keys: [KEY.LEFT,  KEY.A], mode: 'down', action: function() { keyLeft   = true;  } },
      { keys: [KEY.RIGHT, KEY.D], mode: 'down', action: function() { keyRight  = true;  } },
      { keys: [KEY.UP,    KEY.W], mode: 'down', action: function() { keyFaster = true;  } },
      { keys: [KEY.DOWN,  KEY.S], mode: 'down', action: function() { keySlower = true;  } },
      { keys: [KEY.LEFT,  KEY.A], mode: 'up',   action: function() { keyLeft   = false; } },
      { keys: [KEY.RIGHT, KEY.D], mode: 'up',   action: function() { keyRight  = false; } },
      { keys: [KEY.UP,    KEY.W], mode: 'up',   action: function() { keyFaster = false; } },
      { keys: [KEY.DOWN,  KEY.S], mode: 'up',   action: function() { keySlower = false; } }
    ],

    ready: function(images) {
      background = images[0];
      sprites    = images[1];
      reset();
      Dom.storage.fast_lap_time = Dom.storage.fast_lap_time || 180;
      updateHud('fast_lap_time', formatTime(Util.toFloat(Dom.storage.fast_lap_time)));
    },

    afterUpdate: function(){ 
      // if collision or off-road occurs, restart the game
      var pos = Math.abs(playerX);
      if (COLLISION_OCCURED || pos > 0.8){
        TERMINAL = true;
        var json = capture();
        socket.emit('message', json);
        Game.restart();
      }
    }
  };








})();