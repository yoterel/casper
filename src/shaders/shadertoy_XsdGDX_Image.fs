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
const float c_halfGridSize = c_gridSize * 0.5;
const float c_thirdGridSize = c_gridSize / 3.0;
const float c_quarterGridSize = c_gridSize * 0.25;
const float c_maxGridCell = c_gridSize - 1.0;
const float c_radius = 1.0/c_gridSize;

const vec2 txApple = vec2(3.0, c_gridSize);  // x,y = location of apple. z = apple is spawned. w unused

//=======================================================================================
// Debug Visualizations

// #define DV_PLAYBOX  0        // visualize the box of play - where the snake and apples are
// #define DV_PLAYGRID 0        // visualize the 2d grid in the box of play.

//============================================================

// save/load code from IQ's shader: https://www.shadertoy.com/view/MddGzf

vec4 loadValue( in vec2 re )
{
    return texture( iChannel0, (0.5+re) / iResolution.xy, -100.0 );
}

//=======================================================================================
bool RayIntersectAABox (vec3 boxMin, vec3 boxMax, in vec3 rayPos, in vec3 rayDir, out vec2 time)
{
	vec3 roo = rayPos - (boxMin+boxMax)*0.5;
    vec3 rad = (boxMax - boxMin)*0.5;

    vec3 m = 1.0/rayDir;
    vec3 n = m*roo;
    vec3 k = abs(m)*rad;
	
    vec3 t1 = -n - k;
    vec3 t2 = -n + k;

    time = vec2( max( max( t1.x, t1.y ), t1.z ),
                 min( min( t2.x, t2.y ), t2.z ) );
	
    return time.y>time.x && time.y>0.0;
}

//=======================================================================================
bool RayIntersectAABoxNormal (vec3 boxMin, vec3 boxMax, in vec3 rayPos, in vec3 rayDir, out vec3 hitPos, out vec3 normal, inout float maxTime)
{
    vec3 boxCenter = (boxMin+boxMax)*0.5;
	vec3 roo = rayPos - boxCenter;
    vec3 rad = (boxMax - boxMin)*0.5;

    vec3 m = 1.0/rayDir;
    vec3 n = m*roo;
    vec3 k = abs(m)*rad;
	
    vec3 t1 = -n - k;
    vec3 t2 = -n + k;

    vec2 time = vec2( max( max( t1.x, t1.y ), t1.z ),
                 min( min( t2.x, t2.y ), t2.z ) );
    
    // if the time is beyond the maximum allowed bail out (we hit somethign else first!)
    if (time.x > maxTime)
        return false;
    
    // if time invalid or we hit from inside, bail out
    if (time.y < time.x || time.x < 0.0)
        return false;
	
    // calculate surface normal
    hitPos = rayPos + rayDir * time.x;   
    vec3 hitPosRelative = hitPos - boxCenter;
    vec3 hitPosRelativeAbs = abs(hitPosRelative);
    vec3 distToEdge = abs(hitPosRelativeAbs - rad);

    float closestDist = 1000.0;
    for(int axis = 0; axis < 3; ++axis)
    {
        if (distToEdge[axis] < closestDist)
        {
            closestDist = distToEdge[axis];
            normal = vec3(0.0);
            if (hitPosRelative[axis] < 0.0)
                normal[axis] = -1.0;
            else
                normal[axis] = 1.0;
        }
    }        

    // store the collision time as the new max time
    maxTime = time.x;
    return true;
}

//=======================================================================================
bool RayIntersectSphere (in vec4 sphere, in vec3 rayPos, in vec3 rayDir, out vec3 normal, out vec3 hitPos, inout float maxTime)
{
    if (sphere.w <= 0.0)
        return false;
    
	//get the vector from the center of this circle to where the ray begins.
	vec3 m = rayPos - sphere.xyz;

    //get the dot product of the above vector and the ray's vector
	float b = dot(m, rayDir);

	float c = dot(m, m) - sphere.w * sphere.w;

	//exit if r's origin outside s (c > 0) and r pointing away from s (b > 0)
	if(c > 0.0 && b > 0.0)
		return false;

	//calculate discriminant
	float discr = b * b - c;

	//a negative discriminant corresponds to ray missing sphere
	if(discr < 0.0)
		return false;

	//ray now found to intersect sphere, compute smallest t value of intersection
	float collisionTime = -b - sqrt(discr);

	//if t is negative, ray started inside sphere so clamp t to zero and remember that we hit from the inside
	if(collisionTime < 0.0)
        return false;
    
    // if the time is beyond the maximum allowed bail out (we hit somethign else first!)
    if (collisionTime > maxTime)
        return false;
    
    // store the collision time as the new max time
    maxTime = collisionTime;
    
	//compute the point of intersection
	hitPos = rayPos + rayDir * collisionTime;    
    
   	// calculate the normal
	normal = hitPos - sphere.xyz;
	normal = normalize(normal); 

	return true;
}

//=======================================================================================
bool RayIntersectSnakeBody (in vec3 rayPos, in vec3 rayDir, out vec3 normal, out vec3 hitPos, inout float maxTime, out vec3 diffuseColor, out float shinyness)
{
	const float c_radius = 1.0/c_gridSize;
    bool hit = false;
    
    // find where the ray starts and ends within the grid, and then walk the grid to
    // find cells to test against.
    // grid cell walking from http://www.cse.yorku.ca/~amana/research/grid.pdf
    vec2 hitTime;
    if (!RayIntersectAABox(vec3(-1.0,0.0,-1.0), vec3(1.0,c_radius*2.0,1.0), rayPos, rayDir, hitTime))
        return false;
       
    // Grid traversal from http://www.cse.yorku.ca/~amana/research/grid.pdf
    vec3 lineStart = rayPos+(rayDir*hitTime.x);
    vec3 lineEnd = rayPos+(rayDir*hitTime.y);
    
    // calculate where the line starts and stops in grid coordinates
    vec2 lineStartGrid = max(min((lineStart.xz+1.0)*c_halfGridSize, c_gridSize-0.001),0.0);
    vec2 lineEndGrid = max(min((lineEnd.xz+1.0)*c_halfGridSize, c_gridSize-0.001),0.0);
    
    // calculate where the cell positions that the line starts and stops at
    vec2 lineStartCell = floor(lineStartGrid);
    vec2 lineEndCell = floor(lineEndGrid);
    
    // step direction of traversal on each axis
    float stepX = lineEndGrid.x > lineStartGrid.x ? 1.0 : -1.0;
    float stepY = lineEndGrid.y > lineStartGrid.y ? 1.0 : -1.0;
    
    // how far to edge on each axis
    vec2 len = abs(lineEndGrid - lineStartGrid);

    // tDelta.x is time t that it takes to cross a cell horizontally (to next X cell)
    // tDelta.y is time t that it takes to cross a cell vertically (to next Y cell)
    vec2 tDelta = 1.0 / len;
    
    // tMaxX is time t when the line hits a vertical line in the grid (the next X cell)
    // tMaxY is time t when the line hits a horizontal line in the grid (the next Y cell)
    vec2 frac = fract(lineStartGrid);
    float tMaxX = lineEndGrid.x > lineStartGrid.x ? (1.0 - frac.x) / len.x : frac.x / len.x;
    float tMaxY = lineEndGrid.y > lineStartGrid.y ? (1.0 - frac.y) / len.y : frac.y / len.y; 
    
    vec4 cell = texture( iChannel0, (lineStartCell+0.5) / iResolution.xy, -100.0 );
    float offsetY = sin(iTime*5.0) * (1.0 - cell.z) * c_radius / 3.0;
    vec4 sphere;
    sphere.x = (lineStartCell.x-c_halfGridSize)/c_halfGridSize + c_radius;
    sphere.y = c_radius + offsetY;
    sphere.z = (lineStartCell.y-c_halfGridSize)/c_halfGridSize + c_radius;
    sphere.w = c_radius * cell.x;

    if (RayIntersectSphere(sphere, rayPos, rayDir, normal, hitPos, maxTime))
    {
        hit = true;
        diffuseColor = vec3(cell.y,cell.z,0.0);
        shinyness = 0.5;
        return true;
    }    
    
    // Loop
    const int c_loopCount = int(c_gridSize*2.0);
    for (int i = 0; i < c_loopCount; ++i)
    {
        if (tMaxX < tMaxY)
        {
            tMaxX += tDelta.x;
            lineStartCell.x += stepX;
        }
        else
        {
            tMaxY += tDelta.y;
            lineStartCell.y += stepY;
        }
        
        // bail out if we are out of bounds
        if (lineStartCell.x < 0.0 || lineStartCell.y < 0.0
           || lineStartCell.x > c_maxGridCell || lineStartCell.y > c_maxGridCell)
        {
            break;
        }        
        
        vec4 cell = texture( iChannel0, (lineStartCell+0.5) / iResolution.xy, -100.0 );
        float offsetY = sin(iTime*5.0) * (1.0 - cell.z) * c_radius / 3.0;
        vec4 sphere;
        sphere.x = (lineStartCell.x-c_halfGridSize)/c_halfGridSize + c_radius;
        sphere.y = c_radius + offsetY;
        sphere.z = (lineStartCell.y-c_halfGridSize)/c_halfGridSize + c_radius;
        sphere.w = c_radius * cell.x;
        
        if (RayIntersectSphere(sphere, rayPos, rayDir, normal, hitPos, maxTime))
		{
            hit = true;
            diffuseColor = vec3(cell.y,cell.z,0.0);
            shinyness = 0.5;                          
            return true;
		}            
        
        // bail out if we are done
        if (lineStartCell == lineEndCell)
        {     
            break;
        }
    }
    
    return hit;
}

//=======================================================================================
void TraceRay (in vec3 rayPos, in vec3 rayDir, inout vec3 pixelColor, in vec4 apple)
{
    vec3 reverseLightDir = normalize(vec3(-1.0,3.0,-1.0));
    const vec3 lightColor = vec3(0.5,0.5,0.5);	
    const vec3 ambientColor = vec3(0.2,0.2,0.2); 
    
	vec3 normal = vec3(0.0);            
    vec3 diffuseColor;
    float shinyness = 0.0;
    float maxRayHitTime = 1000.0;
    bool hit = false;
    vec3 hitPos = vec3(0.0);
    
    //----- Ray vs snake
    if (RayIntersectSnakeBody(rayPos, rayDir, normal, hitPos, maxRayHitTime, diffuseColor, shinyness))
        hit = true;
    
	//----- Ray vs game board
    if (RayIntersectAABoxNormal(vec3(-1.0,-0.2,-1.0), vec3(1.0,0.0,1.0), rayPos, rayDir, hitPos, normal, maxRayHitTime))
    {
        hit = true;
        // wooden game board framing
        if (hitPos.x > 0.98 || hitPos.x < -0.98 || hitPos.z > 0.98 || hitPos.z < -0.98 || hitPos.y < -0.01)
        {       
            diffuseColor = vec3(1.0, 1.0, 1.0);
            shinyness = diffuseColor.r * 0.25;
        }
        // tiled game board
        else
        {
            bool tileIsWhite = mod(floor(hitPos.x * c_quarterGridSize) + floor(hitPos.z * c_quarterGridSize), 2.0) < 1.0;
            vec3 textureSample = vec3(1.0, 1.0, 1.0);
            shinyness = (tileIsWhite ? 1.0 : 0.5) * textureSample.r;
            diffuseColor = mix(vec3(tileIsWhite ? 1.0 : 0.4), textureSample, 0.33);        
        }
    }
	
    if (!hit)
		return;
    
    // directional light diffuse
    pixelColor = diffuseColor * ambientColor;
    float dp = dot(normal, reverseLightDir);
	if(dp > 0.0)
		pixelColor += (diffuseColor * dp * lightColor);    
    
    // directional light specular    
    vec3 reflection = reflect(reverseLightDir, normal);
    dp = dot(rayDir, reflection);
    if (dp > 0.0)
        pixelColor += pow(dp, 15.0) * 0.5 * shinyness;	
    
    // reflection (environment mappping)
    reflection = reflect(rayDir, normal);
    //pixelColor += texture(iChannel3, reflection).rgb * 0.5 * shinyness;
    
    // handle point light at apple
    if (apple.z == 1.0)
    {
        const vec3 appleLightColor = vec3(5.0,0.0,0.0);
        vec2 appleGrid = floor(clamp(apple.xy * c_maxGridCell, 0.0, c_maxGridCell));
        float offsetY = sin(iTime*5.0) * c_radius / 3.0;
        vec3 applePos;
        applePos.x = (appleGrid.x-c_halfGridSize)/c_halfGridSize + c_radius;
        applePos.y = c_radius + offsetY;
        applePos.z = (appleGrid.y-c_halfGridSize)/c_halfGridSize + c_radius;
        
        float dist = length(applePos - hitPos);
        float distFactor = 1.0 - clamp(dist / 1.0, 0.0, 1.0);
        distFactor = pow(distFactor, 5.0);
        
		// diffuse
		vec3 hitToLight = normalize(applePos - hitPos);
		float dp = dot(normal, hitToLight);
		if(dp > 0.0)
			pixelColor += diffuseColor * dp * appleLightColor * distFactor;
			
		// specular
		vec3 reflection = reflect(hitToLight, normal);
		dp = dot(rayDir, reflection);
		if (dp > 0.0)
			pixelColor += pow(dp, 15.0) * appleLightColor * shinyness * distFactor;
    }    
}

//=======================================================================================
void main()
{
    // load the location of the apple, so  we can use it for a dynamic point light
    vec4 apple = loadValue(txApple);
    
    //----- camera
    vec2 mouse = vec2(0.5,0.5);

    vec3 cameraAt 	= vec3(0.0,0.0,0.0);

    float angleX = 0.0;
    float angleY = 0.0;

    vec3 cameraPos	= (vec3(sin(angleX)*cos(angleY), sin(angleY), cos(angleX)*cos(angleY))) * 4.0;

    vec3 cameraFwd  = normalize(cameraAt - cameraPos);
    vec3 cameraLeft  = normalize(cross(normalize(cameraAt - cameraPos), vec3(0.0,sign(cos(angleY)),0.0)));
    vec3 cameraUp   = normalize(cross(cameraLeft, cameraFwd));

    float cameraViewWidth	= 6.0;
    float cameraViewHeight	= cameraViewWidth * iResolution.y / iResolution.x;
    float cameraDistance	= 6.0;  // intuitively backwards!    
    
	vec2 rawPercent = (fragCoord.xy / iResolution.xy);
	vec2 percent = rawPercent - vec2(0.5,0.5);
	
	vec3 rayPos;
	vec3 rayTarget;
	
	// if the mouse button is down
	
	// else handle the case of the mouse button not being down
	
		rayPos = vec3(0.0,3.0,-3.0);
		vec3 f = normalize(cameraAt - rayPos);
		vec3 l = normalize(cross(f,vec3(0.0,1.0,0.0)));
		vec3 u = normalize(cross(l,f));
		
		rayTarget = (f * cameraDistance)
				  + (l * percent.x * cameraViewWidth)
		          + (u * percent.y * cameraViewHeight);		

	
	vec3 rayDir = normalize(rayTarget);
    
    //vec3 pixelColor = texture(iChannel0, rawPercent).rbg;
    vec3 pixelColor = vec3(0.0, 0.0, 0.0);
    
	TraceRay(rayPos, rayDir, pixelColor, apple);
    
	fragColor = vec4(pixelColor, 1.0);
}