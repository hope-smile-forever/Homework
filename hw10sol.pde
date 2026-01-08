//goal: 使用processing模擬如下簡諧運動(simple harmonic motion).

float amplitude = 100; // 振幅
float omega = 0.05;    // 角頻率 
float phase = 0;       // 相角
float x = 0;           // 當前位移
float time = 0;        // 模擬時間
float L0 = 120;  


int springStart = -200; // 彈簧起點

void setup() {
  size(600, 400);
  frameRate(60);
}

void drawSpring(float xEnd) {
  stroke(0);
  strokeWeight(2);
  noFill();

  int coils = 10;      // 鋸齒段數
  float zigAmp = 18;   // 鋸齒高度
  float hook = 12;     // 兩端小直線
  float minLen = 40;   // 避免太短變形


  if (xEnd < minLen) xEnd = minLen;

  float startX = springStart;
  float endX   = springStart + xEnd;

  // 兩端掛鉤
  float sx = startX + hook;
  float ex = endX - hook;

  // 固定端到彈簧本體
  line(startX, 0, sx, 0);

  // 彈簧本體（鋸齒折線）
  beginShape();
  vertex(sx, 0);

  float seg = (ex - sx) / coils;
  for (int i = 1; i < coils; i++) {
    float px = sx + seg * i;
    float py = (i % 2 == 0) ? -zigAmp : zigAmp;
    vertex(px, py);
  }

  vertex(ex, 0);
  endShape();

  line(ex, 0, endX, 0);
}
void drawWall() {
  stroke(0);
  strokeWeight(3);
  line(springStart -0, -80, springStart - 0, 80);
}

void draw() {
  background(255);
  translate(width/2, height/2);

  x = L0 + amplitude * cos(omega * time + phase);

  drawWall();      // 牆
  drawSpring(x);   // 彈簧
  
  float massX = springStart + x;
  rectMode(CENTER);
  fill(0, 0, 200);
  noStroke();
  rect(massX, 0, 20, 20);

  time += 1;
}
