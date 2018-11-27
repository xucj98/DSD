#ifndef BMP_H
#define BMP_H

#define SHOW_CONSOLE

#pragma pack(1) 

struct BMP_HEADER //λͼͷ�ļ�
{ 
	short bfType; //�ļ����� BM
    int bfSize; //�ļ���С ���ֽڣ�
    short bfReserved1; //����λ 
    short bfReserved2; //����λ
    int bfOffBits; //���ļ�ͷ��ʼ��ͼ����Ϣ��ƫ��λ
};

struct BMP_INFO //λͼ��Ϣ��
{  
    int Size; //λͼ��Ϣ�δ�С
    int Width; //λͼ��ȣ����أ�
    int Height; //λͼ�߶ȣ����أ�
    short Planes; //λͼƽ����
    short BitCount; //˵��ͼƬ�� bit/���� ��С
    int Compression; //����ѹ������
    int SizeImage; //ͼ���С = λͼ��� * λͼ�߶� * Byte/���� 
    int XPelsPerMeter; //ˮƽ�ֱ��ʣ�����/�ף�
    int YPelsPerMeter; //��ֱ�ֱ��ʣ�����/�ף�
    int ClrUsed; //λͼʵ��ʹ�õĲ�ɫ���е���ɫ����������Ϊ0�Ļ�����˵��ʹ�����е�ɫ���
    int ClrImportant; //��ͼ����ʾ����ҪӰ�����ɫ��������Ŀ�������0����ʾ����Ҫ
}; 
struct BMP_PALETTE //��ɫ��
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

	bool file_read(const char *filename); //���ļ�����
	void file_write(const char *filename); //д���ļ�
	int get_color(int x, int y); //��� (x, y) λ�õ���ɫ��ret = (R << 16) | (G << 8) | B;
};

void BMP_output(const BMP_FILE &bmp_file);

#endif