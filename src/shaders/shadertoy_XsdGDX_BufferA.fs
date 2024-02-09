#version 330 core
in vec2 texCoord;
uniform vec2 iResolution;
uniform float iTime;
uniform float iTimeDelta;
uniform int iFrame;
uniform sampler2D iChannel0;
out vec4 fragColor;
vec2 fragCoord = gl_FragCoord.xy;

// the size in X and Y of our gameplay grid
const float c_gridSize = 16.0;
const float c_maxGridCell = c_gridSize - 1.0;

// game speed (snake movement speed) in ticks per second
const float c_tickRate = 1.0;

// The grid representing the board
// x = radius
// y,z = red,green
// w = lifetime in ticks (0..255 -> 0..1)
const vec4 txCells = vec4(0.0, 0.0, c_gridSize - 1.0, c_gridSize - 1.0);

// other variables
const vec2 txPos   = vec2(0.0, c_gridSize);  // x,y = snake head pos.   z,w unused
const vec2 txDir   = vec2(1.0, c_gridSize);  // x,y = snake direction.  z,w -> new desired dir based on key presses.
const vec2 txState = vec2(2.0, c_gridSize);  // x = state. y = percent til tick. z = snake length (0.255 -> 0..1). w unused.
const vec2 txApple = vec2(3.0, c_gridSize);  // x,y = location of apple. z = apple is spawned. w unused

// keys
const float KEY_SPACE = 32.5/256.0;
const float KEY_LEFT  = 37.5/256.0;
const float KEY_UP    = 38.5/256.0;
const float KEY_RIGHT = 39.5/256.0;
const float KEY_DOWN  = 40.5/256.0;

//============================================================

// save/load code from IQ's shader: https://www.shadertoy.com/view/MddGzf

float isInside( vec2 p, vec2 c ) { vec2 d = abs(p-0.5-c) - 0.5; return -max(d.x,d.y); }
float isInside( vec2 p, vec4 c ) { vec2 d = abs(p-0.5-c.xy-c.zw*0.5) - 0.5*c.zw - 0.5; return -max(d.x,d.y); }

vec4 loadValue( in vec2 re )
{
    return texture( iChannel0, (0.5+re) / iResolution.xy, -100.0 );
}

void storeValue( in vec2 re, in vec4 va, inout vec4 fragColor, in vec2 fragCoord )
{
    fragColor = ( isInside(fragCoord,re) > 0.0 ) ? va : fragColor;
}

void storeValue( in vec4 re, in vec4 va, inout vec4 fragColor, in vec2 fragCoord )
{
    fragColor = ( isInside(fragCoord,re) > 0.0 ) ? va : fragColor;
}

//============================================================
float rand(vec2 co)
{
    return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
}

//============================================================
void main()
{   
    if (fragCoord.x > c_gridSize || fragCoord.y > c_gridSize + 1.0)
        discard;
    
    //----- Load State -----
    vec2 cellPos = floor(fragCoord);  
    
    vec4 pos   = loadValue(txPos);
    vec4 dir   = loadValue(txDir);
    vec4 state = loadValue(txState);
    vec4 cell  = loadValue(fragCoord.xy-0.5);
    vec4 apple = loadValue(txApple);
    
    // convert dir from 0..1 to -1..1
    dir = (dir * 2.0) - 1.0;
    
    // reset game state on first frame
    if (iFrame == 0)
        state = vec4(0.0);
    
    // state .0 -> reset game state
    if (state.x < 0.1)
    {
        state.x = 0.1;
        state.y = 0.0;
        state.z = 3.0 / 255.0;
        pos   = vec4(0.5, 0.5, 0.0, 0.0);
        dir   = vec4(0.0);
        apple = vec4(0.0);
        
        // write a snake body at the starting position of the player
        vec2 posGrid = floor(pos.xy * c_maxGridCell);
        if (cellPos == posGrid)
            cell = vec4(1.0,0.0,1.0,state.z);
        else
            cell = vec4(0.0);
    }
   	// state .1 -> we are playing!
    else if (state.x < 0.2)
    {        
        // try and spawn the apple if we need to
        if (apple.z == 0.0)
        {
            //float test = 0.5;
            apple.x = rand(vec2(float(iFrame)*0.122+float(iFrame)*0.845, float(iFrame)*0.647+float(iFrame)*0.753));
            apple.y = rand(vec2(float(iFrame)*0.546+float(iFrame)*0.342, float(iFrame)*0.342+float(iFrame)*0.935));
            vec2 appleCell = floor(apple.xy * c_maxGridCell);
            
            // if we fail to select an empty spot on the grid, try again next frame.
            vec4 cellAtApple = texture( iChannel0, (appleCell+0.5) / iResolution.xy, -100.0 );
            if (cellAtApple.z == 0.0)
            {
                apple.z = 1.0;
                
                if (cellPos == appleCell)
            		cell = vec4(0.75,1.0,0.0,0.0);
            }
        }
        
        // handle queueing up directional changes
        float moveLeft  = 0.0; //texture( iChannel1, vec2(KEY_LEFT,0.25) ).x;            
        float moveRight = 1.0; //texture( iChannel1, vec2(KEY_RIGHT,0.25) ).x;
        float moveUp    = 0.0; //texture( iChannel1, vec2(KEY_UP,0.25) ).x;
        float moveDown  = 0.0; //texture( iChannel1, vec2(KEY_DOWN,0.25) ).x;

        if (moveLeft == 1.0 && dir.x == 0.0)
        {
            dir.z =  1.0;
            dir.w =  0.0;
        }
        else if (moveRight == 1.0 && dir.x == 0.0)
        {
            dir.z = -1.0;
            dir.w =  0.0;
        }
        else if (moveUp == 1.0 && dir.y == 0.0)
        {
            dir.z =  0.0;
            dir.w =  1.0;                
        }
        else if (moveDown == 1.0 && dir.y == 0.0)
        {
            dir.z =  0.0;
            dir.w = -1.0;                
        }        
        
        // tick() when we should
        state.y += iTimeDelta * c_tickRate;
        if (state.y > 1.0)
        {
            bool ateApple = false;
            
            state.y = fract(state.y);
            
        	// handle queued direction changes
            dir.xy = dir.zw;
            
            // if the snake is moving
            if (length(dir) > 0.0)
            {
                // handle snake movement
                vec2 posGrid = floor(pos.xy * c_maxGridCell);
                posGrid += dir.xy;
                pos.xy = posGrid / c_maxGridCell;

                // you die if you go out of bounds
                if (posGrid.x < 0.0 || posGrid.y < 0.0 || posGrid.x > c_maxGridCell || posGrid.y > c_maxGridCell)
                    state.x = 0.2;         
                
                vec4 cellAtSnakeHead = texture( iChannel0, (posGrid+0.5) / iResolution.xy, -100.0 );

                // you also die if the new place you want to go already has a snake in it
                if (length(dir) > 0.0 && texture( iChannel0, (posGrid+0.5) / iResolution.xy, -100.0 ).x == 1.0)
                    state.x = 0.2;
                
                // if you eat an apple, we need to spawn a new apple and also increment snake lifetime in state
                if (cellAtSnakeHead.y == 1.0)
                {
                    apple.z = 0.0;
                    state.z = (floor(state.z * 255.0) + 1.0) / 255.0;
                    ateApple = true;
                }                         
                
                // if the cell we are processing is the new head of the snake, put a snake body part there
                if (cellPos == posGrid)
                    cell = vec4(1.0,0.0,1.0, state.z); 
				// if we didn't eat an apple this tick, decriment lifetime of all snake body parts
                // and destroy any that hit zero
                else if (!ateApple && cell.w > 0.0)
                {
                    cell.w = (floor(cell.w * 255.0) - 1.0) / 255.0;
                    if (cell.w <= 0.0)
                        cell = vec4(0.0);
                }
            }
        }
    }
   	// state .2 -> we are dead!
    else if (state.x < 0.3)
    {
        // reset when user presses space
        // if (texture( iChannel1, vec2(KEY_SPACE,0.25) ).x == 1.0)
        	// state.x = 0.0;
        
        // Death effect - make snake yellow
        state.y += iTimeDelta * 30.0;
        if (state.y > 1.0)
        {
            if (cell.x > 0.9)
                cell.y = min(cell.y+0.25,1.0);
            
            state.y = 0.0;
        }
    }
    
    // convert dir from -1..1 to 0..1
    dir = (dir + 1.0) * 0.5;    
    
    //----- Save State -----
    fragColor = vec4(0.0);
    storeValue(txPos  , pos  , fragColor, fragCoord);
    storeValue(txDir  , dir  , fragColor, fragCoord);
    storeValue(txState, state, fragColor, fragCoord);
    storeValue(txCells, cell , fragColor, fragCoord);
    storeValue(txApple, apple, fragColor, fragCoord);
}