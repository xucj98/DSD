#ifndef BMP_H
#define BMP_H

#define SHOW_CONSOLE

#pragma pack(1) 

struct BMP_HEADER //位图头文件
{ 
	short bfType; //文件类型 BM
    int bfSize; //文件大小 （字节）
    short bfReserved1; //保留位 
    short bfReserved2; //保留位
    int bfOffBits; //从文件头开始到图像信息的偏移位
};

struct BMP_INFO //位图信息段
{  
    int Size; //位图信息段大小
    int Width; //位图宽度（像素）
    int Height; //位图高度（像素）
    short Planes; //位图平面属
    short BitCount; //说明图片的 bit/像素 大小
    int Compression; //数据压缩类型
    int SizeImage; //图像大小 = 位图宽度 * 位图高度 * Byte/像素 
    int XPelsPerMeter; //水平分辨率（像素/米）
    int YPelsPerMeter; //垂直分辨率（像素/米）
    int ClrUsed; //位图实际使用的彩色表中的颜色索引数（设为0的话，则说明使用所有调色板项）
    int ClrImportant; //对图象显示有重要影响的颜色索引的数目，如果是0，表示都重要
}; 
struct BMP_PALETTE //调色板
{  
	unsigned char peRed; 
    unsigned char peGreen; 
    unsigned char peBlue; 
    unsigned char peFlags; 
}; 

class BMP_FILE
{
public:
	BMP_HEADER header;
    BMP_INFO info;
	BMP_PALETTE palette[256];
    unsigned char *buffer;   

	BMP_FILE();
	~BMP_FILE();

	bool file_read(const char *filename); //从文件读入
	void file_write(const char *filename); //写入文件
	int get_color(int x, int y); //获得 (x, y) 位置的颜色，ret = (R << 16) | (G << 8) | B;
};

void BMP_output(const BMP_FILE &bmp_file);

#endif