#include "BMP.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
using namespace std;

BMP_FILE::BMP_FILE()
{
	buffer = NULL;
}

BMP_FILE::~BMP_FILE()
{
	free(buffer);
	buffer = NULL;
}

bool BMP_FILE::file_read(const char *filename)
{
	FILE *fp = fopen(filename, "rb");
	if (fp == NULL) return false;
	fread(&header, sizeof(header), 1, fp);
	fread(&info, sizeof(info), 1, fp);
	int lineByte=(info.Width * info.BitCount/8+3)/4*4;
	if (buffer != NULL) free(buffer);
	buffer = (unsigned char *) malloc(info.Height * lineByte);
	fseek(fp, header.bfOffBits, SEEK_SET);
	fread(buffer, info.Height * lineByte, 1, fp);
	fclose(fp);
	return true;
}

void BMP_FILE::file_write(const char *filename)
{
	FILE *fp = fopen(filename, "wb");
	fwrite(&header, sizeof(header), 1, fp);
	fwrite(&info, sizeof(info), 1, fp);
	int lineByte=(info.Width * info.BitCount/8+3)/4*4;
	fwrite(buffer, info.Height * lineByte, 1, fp);
	fclose(fp);
}

int BMP_FILE::get_color(int x, int y)
{
    y = info.Height - y - 1;
	int lineByte=(info.Width * info.BitCount/8+3)/4*4;
	unsigned char *tmp = buffer + y*lineByte + x*3;
	return *tmp | (*(tmp+1) << 8) | (*(tmp + 2) << 16);
}

void BMP_output(const BMP_FILE &bmp_file)
{
	printf("BMP_HEADER:\n");
	printf("	bfType: %x\n", bmp_file.header.bfType);
	printf("	bfSize: %d\n", bmp_file.header.bfSize);
	printf("	bfReserved1: %d\n", bmp_file.header.bfReserved1);
	printf("	bfReserved2: %d\n", bmp_file.header.bfReserved2);
	printf("	bfOffBits: %d\n", bmp_file.header.bfOffBits);
	printf("\n");

	printf("BMP_INFO:\n");
	printf("	Size: %d\n", bmp_file.info.Size);
	printf("	Width: %d\n", bmp_file.info.Width);
	printf("	Height: %d\n", bmp_file.info.Height);
	printf("	Planes: %d\n", bmp_file.info.Planes);
	printf("	BitCount: %d\n", bmp_file.info.BitCount);
	printf("	Compression: %d\n", bmp_file.info.Compression);
	printf("	SizeImage: %d\n", bmp_file.info.SizeImage);
	printf("	XPelsPerMeter: %d\n", bmp_file.info.XPelsPerMeter);
	printf("	YPelsPerMeter: %d\n", bmp_file.info.YPelsPerMeter);
	printf("	ClrUsed: %d\n", bmp_file.info.ClrUsed);
	printf("	ClrImportant: %d\n", bmp_file.info.ClrImportant);
}

