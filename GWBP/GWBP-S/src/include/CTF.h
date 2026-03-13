#ifndef CTF_H 
#define CTF_H 

class CTF
{
public:
    CTF();
    CTF(int n_now);
    ~CTF();

public:
    // 设置CTF相关参数
    void setAllCTFPara(float defocus1_now, float defocus2_now, float astig_now, float phase_shift_now, float w_now);
    // 设置图像相关参数
    void setAllImagePara(float pix_now, float volt_now, float Cs_now);
    // 设置n值
    void setN(int n_now);

    // 获取相关参数的getter方法
    float getDefocus1();
    float getDefocus2();
    float getAstigmatism();
    float getPhaseShift();
    float getWcos();
    float getWsin();
    float getPixelSize();
    float getVoltage();
    float getCs();
    float getLambda();
    int getN();
    // 新增：声明getW()函数（对应cpp中的实现）
    float getW();

    // 计算2D CTF（原有函数，保持不变）
    float computeCTF2D(float x, float y, int Nx, int Ny, bool phaseflip, bool flip_contrast, float z_offset);
    
    // 修改：computeCTF2D_P声明，与cpp实现的参数完全匹配
    float computeCTF2D_P(float &defocus1_p, float &defocus2_p, float &astig_p, 
                       float &phase_shift_p, float &w_cos_p, float &w_sin_p, 
                       float &pix_p, float &pix_1_p, float &lambda_p, 
                       float &lambda2_p, float &lambda3_p);

//private:
public:
    // 原有成员变量
    float defocus1, defocus2;    // 焦距参数，单位：m
    float astig;                 // 像散，单位：弧度
    float phase_shift;           // 相移，单位：弧度
    float w_cos, w_sin;          // 权重相关参数
    float pix;                   // 像素尺寸，单位：m
    float volt;                  // 电压，单位：V
    float Cs;                    // 球差系数，单位：m
    float lambda;                // 波长，单位：m
    int n;                       // 相关整数参数

    // 新增：补充cpp中使用但未声明的成员变量
    float pix_1;                 // pix的倒数（1/pix）
    float lambda2;               // M_PI * lambda（CTF计算中间变量）
    float lambda3;               // M_PI_2 * Cs * lambda^3（CTF计算中间变量）
};

#endif