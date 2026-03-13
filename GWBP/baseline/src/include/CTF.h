class CTF
{
public:
    CTF();
    CTF(int n_now);
    ~CTF();

public:
    void setAllCTFPara(float defocus1_now,float defocus2_now,float astig_now,float phase_shift_now,float w_now);
    void setAllImagePara(float pix_now,float volt_now,float Cs_now);
    void setN(int n_now);

    float getDefocus1();
    float getDefocus2();
    float getAstigmatism();
    float getPhaseShift();
    float getW();
    float getPixelSize();
    float getVoltage();
    float getCs();
    float getLambda();
    int getN();

    float computeCTF2D(float x,float y,int Nx,int Ny,bool phaseflip,bool flip_contrast,float z_offset);
    float computeCTF2D_P(float &defocus1_p,float &defocus2_p,float &astig_p,float &phase_shift_p,float &w_cos_p,float &w_sin_p,float &pix_p, float &pix_1_p, float &lambda_p, float &lambda2_p, float &lambda3_p);

private:
    float defocus1,defocus2;    // in m
    float astig;    // in radius
    float phase_shift;  // in radius
    float w_cos,w_sin;

    float pix;  // in m
	float pix_1;
    float volt; // in V
    float Cs;   // in m
    float lambda;   // in m
	float lambda3;
	float lambda2;
    int n;

};
