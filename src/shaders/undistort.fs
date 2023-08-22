float4 CorrectBarrelDistortion(float2 fptexCoord : TEXCOORD0,
                               float4 wpos : WPOS,
                                uniform samplerRECT FPE1 ) : COLOR 
{    
    // known constants for the particular lens and sensor
    float f = 368.28488;// focal length 
    float ox = 147.54834; // principal point, x axis
    float oy = 126.01673; // principal point, y axis 
    float k1 = 0.4142;    // constant for radial distortion correction
    float k2 = 0.40348;
    float2 xy = (fptexCoord - float2(ox, oy))/float2(f,f);
    r = sqrt( dot(xy, xy) );
    float r2 = r * r;
    float r4 = r2 * r2;
    float coeff = (k1 * r2 + k2 * r4);    // add the calculated offsets to the current texture coordinates
    xy = ((xy + xy * coeff.xx) * f.xx) + float2(ox, oy);    // look up the texture at the corrected coordinates
    // and output the color 
    return texRECT(FPE1, xy);
} 