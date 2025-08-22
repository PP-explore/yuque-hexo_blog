---
title: C语言学习
date: '2024-03-06 15:26:33'
updated: '2025-08-22 16:22:41'
---
# 存储类型
## C语言每个变量具有三大特性：存储期限、作用域、链接
![画板](/images/b6cc0d2c271a80394e7a0a79ec8d07e5.jpeg)

变量的特性依赖变量声明的位置：

块内部（函数内）声明的变量具有自动存储期限、块作用域、无连接

<font style="color:#DF2A3F;">程序最外层外部声明的变量具有静态存储期限、文件作用域、外部链接</font>

## auto存储类型
只对属于块的变量有效。几乎从不需要明确指明，因为是默认的，具有自动存储期限、块作用域、无连接

## static存储类型
可用于全部变量，无需考虑变量位置

<font style="color:#DF2A3F;">块外声明具有内部链接、静态存储期限、文件作用域</font>

块内部声明则为静态存储期限、块作用域、无连接

### 块内声明的static变量的性质：（信息隐藏技术）
1. <font style="color:#DF2A3F;">块内声明的static只会在程序执行期间进行一次初始化</font>
2. 每次递归调用，所有递归函数都可以共享这个static变量
3. 可以返回指向auto型变量的指针

## extern存储类型
## register存储类型
要求编译器吧变量存储在寄存器中

# 运算符优先级及结合顺序
优先级：按照优先级顺序给子表达式加括号确定编译器解释表达

结合性：含有相同的优先级，则按照结合性确定运算顺序。

双目运算符除了“=”相关运算符，其他都是从左到右结合

| 优先级 | 类型名称 | 符号 | 结合性 |
| --- | --- | --- | --- |
| 1 | <font style="color:rgb(68, 68, 68);">圆括号</font> | <font style="color:rgb(68, 68, 68);">（）</font> | <font style="color:rgb(68, 68, 68);"></font><br/><font style="color:rgb(68, 68, 68);"></font><br/><font style="color:rgb(68, 68, 68);">左到右</font><br/><font style="color:rgb(68, 68, 68);"> </font><br/><font style="color:rgb(68, 68, 68);"> </font> |
| | <font style="color:rgb(68, 68, 68);">数组下标</font> | <font style="color:rgb(68, 68, 68);">【】</font> | |
| | <font style="color:rgb(68, 68, 68);">指向结构体成员选择（指针）</font> | <font style="color:rgb(68, 68, 68);">-></font> | |
| | <font style="color:rgb(68, 68, 68);">结构体成员选择（对象）</font> | <font style="color:rgb(68, 68, 68);">.</font> | |
| | | | |
| 2 | 后缀自增减 | ++、-- | 左结合 |
| 2 | 前缀自增减 | ++、-- | 右结合 |
| 2 |  | & ! ~ * | 右到左 |
| |  |  | |
| 3 | 二元乘除类 | */% | 左结合 |
| 4 | 二元加减类 | +- | 左结合 |
| 5 | & | 位与 | 左结合 |
| 6 | ^ | 位异或 | |
| 7 | | | 位或 | |
| 8 | && | 逻辑与 | |
| 9 | || | 逻辑或 | |
| 10 | 赋值运算符 | =  *+  /+  %=  +=  -= | 右结合 |
| 11 | 逗号运算符 | ， |  |


| <font style="color:rgb(68, 68, 68);">优先级问题</font> | <font style="color:rgb(68, 68, 68);">表达式</font> | <font style="color:rgb(68, 68, 68);">经常误认为的结果</font> | <font style="color:rgb(68, 68, 68);">实际结果</font> |
| --- | --- | --- | --- |
| <font style="color:#DF2A3F;">. 的优先级高于 *（-> 操作符用于消除这个问题）</font> | <font style="color:rgb(68, 68, 68);">*p.f</font> | <font style="color:rgb(68, 68, 68);">p 所指对象的字段 f，等价于：   </font><font style="color:rgb(68, 68, 68);">(*p).f</font> | <font style="color:rgb(68, 68, 68);">对 p 取 f 偏移，作为指针，然后进行解除引用操作，等价于：   </font><font style="color:rgb(68, 68, 68);">*(p.f)</font> |
| <font style="color:rgb(68, 68, 68);">[] 高于 *</font> | <font style="color:rgb(68, 68, 68);">int *ap[]</font> | <font style="color:rgb(68, 68, 68);">ap 是个指向 int 数组的指针，等价于：   </font><font style="color:rgb(68, 68, 68);">int (*ap)[]</font> | <font style="color:rgb(68, 68, 68);">ap 是个元素为 int 指针的数组，等价于：   </font><font style="color:rgb(68, 68, 68);">int *(ap [])</font> |
| <font style="color:rgb(68, 68, 68);">函数 () 高于 *</font> | <font style="color:rgb(68, 68, 68);">int *fp()</font> | <font style="color:rgb(68, 68, 68);">fp 是个函数指针，所指函数返回 int，等价于：   </font><font style="color:rgb(68, 68, 68);">int (*fp)()</font> | <font style="color:rgb(68, 68, 68);">fp 是个函数，返回 int*，等价于：   </font><font style="color:rgb(68, 68, 68);">int* ( fp() )</font> |
| <font style="color:rgb(68, 68, 68);">== 和 != 高于位操作</font> | <font style="color:rgb(68, 68, 68);">(val & mask != 0)</font> | <font style="color:rgb(68, 68, 68);">(val &mask) != 0</font> | <font style="color:rgb(68, 68, 68);">val & (mask != 0)</font> |
| <font style="color:rgb(68, 68, 68);">== 和 != 高于赋值符</font> | <font style="color:rgb(68, 68, 68);">c = getchar() != EOF</font> | <font style="color:rgb(68, 68, 68);">(c = getchar()) != EOF</font> | <font style="color:rgb(68, 68, 68);">c = (getchar() != EOF)</font> |
| <font style="color:#DF2A3F;">算术运算符高于位移 运算符</font> | <font style="color:rgb(68, 68, 68);">msb << 4 + lsb</font> | <font style="color:rgb(68, 68, 68);">(msb << 4) + lsb</font> | <font style="color:rgb(68, 68, 68);">msb << (4 + lsb)</font> |
| <font style="color:#DF2A3F;">逗号运算符在所有运 算符中优先级最低</font> | <font style="color:rgb(68, 68, 68);">i = 1, 2</font> | <font style="color:rgb(68, 68, 68);">i = (1,2)</font> | <font style="color:rgb(68, 68, 68);">(i = 1), 2</font> |


例题：

int i=3;

int m=(++i)-(i++)+(++i);

依照优先级，已经为自运算添加括号，继续为+-添加括号

int m=(((++i)-(i++))+(++i));

由于+-为左结合，同样优先级按照结合性添加括号，从左往右添加

！！但是，一般这个内部自增运算到底按照什么顺序执行，会依赖编译器，不同编译器给的答案不同。

# 低级运算符
移位运算符

<<	左移

>>	右移

~	按位求反

&	按位与

^	按位异或

|	按位或

# 字符串学习
## 字符串数组及字符串指针数组
字符串数组必须在定义的时候赋值，如：

char a[]="Hello";

char a[]={'H','e','l','l','o'};

char *a; a="Hello";//<font style="color:rgb(171, 178, 191);background-color:rgb(40, 44, 52);">等价于 </font><font style="color:rgb(198, 120, 221);">const char </font><font style="color:rgb(102, 153, 0);">*</font><font style="color:rgb(171, 178, 191);background-color:rgb(40, 44, 52);">m</font><font style="color:rgb(102, 153, 0);">="hello"</font><font style="color:rgb(153, 153, 153);">;</font>

char *a[]={"Hello","World","PP"};

第一种直接将字符串赋值，系统会自动在数组末尾添加\0

第二种类似整形 数组赋值，必须自己手动在末尾添加\0

第三种为字符串指针形式,<font style="color:#DF2A3F;">此时*p='A'；为错误写法，因为Hello为常量；</font>

第四种为字符串指针数组

注意：char a[]={"Hello","World","PP"};为错误方式

#### 注意：字符型指针的初始化
```c
char *a; scanf("%s",a);
/*此为错误写法！！
a 作为指针，需要初始化指明地址，此时仍为野指针，而使用scanf无法知到输入的字符串长度。
而char *a; a="Hello";这是给指针变量 s 赋予一个字符串字面值（即字符串常量）
此时需要指出“Hello”字符串常量被存储在静态存储区，并且在编译时就已经有了固定的内存地址。因此才是合法的。
*/
char *a; scanf("%ms",a);  free(a);//方法1
char *a=(char*)malloc(sizeog(char)*100); scanf("%s",a);free(a);//方法2
/*
上方为正确使用scanf的写法,要记得free释放内存！
写法1的scanf中的m为非标准扩展格式说明符，%ms 会自动在读取的字符串末尾添加 \0，确保读取的是一个完整的 C 风格的字符串。这个扩展在某些实现（如 GNU libc）中可用。
*/

    printf("输入字符串：");
    scanf("%ms", &a);
    getchar();
    printf("输入字符串2:");
    scanf("%c%c", &b[0], &b[1]);
/*
    注意此处的getchar（）清除了“\n”符号，防止下方%c输入时把\n读入
*/
```



### strcat（字符数组1，字符数组2）
>> 拼接函数：

。 必须字符串数组12都要以“\0”结尾，否则无法识别拼接位置，地址越界

。 字符数组1的末尾\0会被覆盖，拼接后再添加\0

### strcpy（char *dest，char*src）
>>复制函数：

。源地址必须\0结尾

。将src取代dest的起始位置

### strncpy(char*dest,char*src,size_t n)
>>n位复制函数：

。如果src小于n，其余位置则会填充\0

。如果src大于n，末尾则不会添加\0，所以，dest字符串大小应该大于src，否则就要手动添加\0

### strcmp（字符串1，字符串2）
>>比较函数：

。自左向右逐个字符比较俩者ASCII码大小，直到出现末尾遇到\0

。返回0为相等，小于0为字符串1小于字符串2

### strlen（字符数组）
>>长度函数：

。测量长度，不包括\0

### strlwr（字符数组）和strupr（函数）
>>大小写转换函数：

。strlwr为string to lower转小写

.strupr为string to upper转大写

## 注意：<font style="color:#DF2A3F;">strlen和sizeof的区别</font>
strlen（）是遇到第一个’\0‘后便停止且不包含'\0’的长度

sizeof()是查询其占用的内存大小返回字节数，不在乎是否'\0'

C语言中char占用1字节

## #include<stdio.h>
### puts(const char *star)
>> 向标准输出设备屏幕写入字符串并换行，即自动写一个换行符\n

同printf("%s\n",str);相同

需要遇到\0字符才能停止输出

### fputs(const char *str,FILE *stream)
>>向指定的文件写入字符串（不换行），第一个参数为要写入的字符串，第二个为要写入的文件，可以是stdout；

它同样需要玉带\0字符才停止输出

例如fputs（str，stdout）；

### gets(char *str)
>>从输入缓冲区读取一个字符串，会存储空格，Tab键

遇到换行符则停止读取，并把换行符移除缓冲区

读取字符串结束后会自动添加\0

scanf（）则是先找到一个非空字符读入直到遇到空格换行Tab结束读取，并将换行符保留在缓冲区中，所系可能需要为下次读取清空缓冲区

<font style="color:rgb(199, 37, 78);background-color:rgb(249, 242, 244);">gets()</font><font style="color:rgb(77, 77, 77);">不能指定读取上限，因此容易发生数组边界溢出，造成内存不安全</font>

### char* fgets（char * str, int size, FILE *restrict stream）
>> <font style="color:rgb(77, 77, 77);">从指定输入流读取一行，输入可以是</font><font style="color:rgb(199, 37, 78);background-color:rgb(249, 242, 244);">stdin</font><font style="color:rgb(77, 77, 77);">，也可以是文件流，使用时需要显式指定</font>

1. <font style="color:rgba(0, 0, 0, 0.75);">所有空格、Tab等空白字符均被读取，不忽略。</font>
2. <font style="color:rgba(0, 0, 0, 0.75);">按下回车键时，缓冲区末尾的换行符也被读取，字符串末尾将有一个换行符</font><font style="color:rgb(199, 37, 78);background-color:rgb(249, 242, 244);">\n</font><font style="color:rgba(0, 0, 0, 0.75);">。例如，输入字符串</font><font style="color:rgb(199, 37, 78);background-color:rgb(249, 242, 244);">hello</font><font style="color:rgba(0, 0, 0, 0.75);">，再按下回车，则读到的字符串长度为</font><font style="color:rgb(199, 37, 78);background-color:rgb(249, 242, 244);">6</font><font style="color:rgba(0, 0, 0, 0.75);">。</font>
3. <font style="color:rgb(199, 37, 78);background-color:rgb(249, 242, 244);">fgets()</font><font style="color:rgb(77, 77, 77);">函数会自动在字符串末尾加上</font><font style="color:rgb(199, 37, 78);background-color:rgb(249, 242, 244);">\0</font><font style="color:rgb(77, 77, 77);">结束符。</font>
4. **<font style="color:rgb(77, 77, 77);">fgets()函数的返回值和它第一个参数相同。即读取到数据后存储的容器地址。但是如果读取出错或读取文件时文件为空，则返回一个空指针。</font>**
5. **<font style="color:rgb(0, 0, 255);">通俗来讲的话，fgets()函数的作用就是用来读取一行数据的。但要详细且专业的说的话，fgets()函数的作用可以这么解释：从第三个参数指定的流中读取最多第二个参数大小的字符到第一个参数指定的容器地址中。在这个过程中，在还没读取够第二个参数指定大小的字符前，读取到换行符'\n'或者需要读取的流中已经没有数据了。则提前结束，并把已经读取到的字符存储进第一个参数指定的容器地址中。</font>**
6. **<font style="color:rgb(255, 165, 0);">fgets()函数的最大读取大小是其“第二个参数减1”，这是由于字符串是以’\0’为结束符的，fgets()为了保证输入内容的字符串格式，当输入的数据大小超过了第二个参数指定的大小的时候，fgets()会仅仅读取前面的“第二个参数减1”个字符，而预留1个字符的空间来存储字符串结束符’\0’。</font>**
7. **<font style="color:rgb(255, 165, 0);">在fgets()函数的眼里，换行符’\n’也是它要读取的一个普通字符而已。在读取键盘输入的时候会把最后输入的回车符也存进数组里面，即会把’\n’也存进数组里面，而又由于字符串本身会是以’\0’结尾的。</font>**

### [fgetc](https://so.csdn.net/so/search?q=fgetc&spm=1001.2101.3001.7020)<font style="color:rgb(79, 79, 79);">() & getc()</font>
>>

1. 所有空格、Tab、换行等空白字符，无论在缓冲区开头、中间还是结尾，均会被读取，不忽略。
2. 因为只读取一个字符，所以如果输入多于1个字符（包括换行符），则它们均会残留在缓冲区。具体地说，如果什么字符都不输入，直接按下回车键，则读取到的是换行符\n，缓冲区无任何残留；如果输入一个字符如a，然后按下回车键，则读取到的是字符a，同时换行符\n残留在缓冲区。

a =fgetc(stdin); b =getc(stdin);

## 文件相关函数
### exit（）；
关闭所有文件，终止程序，带用户检查错误

### <font style="color:rgb(0, 0, 0);background-color:rgb(250, 250, 250);">FILE </font><font style="color:rgb(166, 127, 89);">*</font><font style="color:rgb(221, 74, 104);">fopen</font><font style="color:rgb(153, 153, 153);">(</font><font style="color:rgb(0, 119, 170);">constchar</font><font style="color:rgb(166, 127, 89);">*</font><font style="color:rgb(0, 0, 0);background-color:rgb(250, 250, 250);"> filename</font><font style="color:rgb(153, 153, 153);">,</font><font style="color:rgb(0, 119, 170);">constchar</font><font style="color:rgb(166, 127, 89);">*</font><font style="color:rgb(0, 0, 0);background-color:rgb(250, 250, 250);"> mode </font><font style="color:rgb(153, 153, 153);">);</font>
### <font style="color:rgb(0, 119, 170);">int </font><font style="color:rgb(221, 74, 104);">fclose</font><font style="color:rgb(153, 153, 153);">(</font><font style="color:rgb(0, 0, 0);background-color:rgb(250, 250, 250);"> FILE </font><font style="color:rgb(166, 127, 89);">*</font><font style="color:rgb(0, 0, 0);background-color:rgb(250, 250, 250);"> stream </font><font style="color:rgb(153, 153, 153);">);</font>
<font style="color:rgb(77, 77, 77);">打开方式：</font>

<font style="color:rgb(77, 77, 77);">"r"	只读		打开文本文件	不存在文件则出错</font>

<font style="color:rgb(77, 77, 77);">"w"	只写		打开文本文件	不存在则新建文件</font>

<font style="color:rgb(77, 77, 77);">"a"	追加		向文本文件添加数据		不存在则新建</font>

<font style="color:rgb(77, 77, 77);">"rb"	只读		打开二进制文件	不存在则出错</font>

<font style="color:rgb(77, 77, 77);">"wb"	 只写	打开二进制文件	不存在则新建文件</font>

"ab"	 追加	二进制文件添加数据		不存在则新建

注意：

fopen要检查是否返回的是空指针，判断是否打开成功

if(fp=fopen("name","rb")==null){printf("打开失败”)；exit(1);}

### fread（buffer，size,count,fp）;fwrite(buffer,size,count,fp);
**<font style="color:rgb(77, 77, 77);">（1）buffer：是一个指针，对fread来说，它是读入数据的存放地址。对fwrite来说，是要输出数据的地址。</font>**

**<font style="color:rgb(77, 77, 77);">（2）size：要读写的字节数；</font>**

**<font style="color:rgb(77, 77, 77);">（3）count:要进行读写多少个size字节的数据项；</font>**

**<font style="color:rgb(77, 77, 77);">（4）fp:文件型指针。</font>**

（5）返回值为实际读写的字节数，若等于第三个参数则没有出错，若不等于则出错

<font style="color:rgb(77, 77, 77);background-color:#FBE4E7;">注意：</font>

1. <font style="color:rgb(77, 77, 77);">完成次写操(fwrite())作后必须关闭流(fclose());</font>
2. <font style="color:rgb(77, 77, 77);">完成一次读操作(fread())后，如果没有关闭流(fclose()),则指针(FILE * fp)自动向后移动前一次读写的长度，不关闭流继续下一次读操作则接着上次的输出继续输出;</font>
3. <font style="color:rgb(77, 77, 77);">一定要检查返回值是否等于第三个参数</font>

### 文件状态：feof和ferror
feof判断是否到达文件末尾，到达则返回非0，未到达为0

ferror检查上一次流操作是否错误，一般形式为ferror（fp）；若返回0则未出错，非0则出错

### 文件输入输出fscanf和fprintf
fprintf(文件指针，格式字符串，输出列表)  

>>输出数据到文件中

fscanf（文件指针 ，格式字符串，输出列表）

>>读取文件中数据到指定变量中

### 文件定位：rewind，fseek，ftell
rewind(文件指针)；是指针重新返回到文件开头，无返回值

fseek(文件指针，字节位移量，起始点)；以起始点为基准向前移动的字节数

0或者SEEK_SET表示文件开始

1或者SEEK_CUR表示当前位置

2或者SEEK_END表示文件末尾

long int ftell（文件指针）

>>以长整型返回当前文件位置，只有在二进制文件中，返回的是从文件开始到当前位置的字节计数。所以可以用来用来判断二进制文件的当前位置

注意：当为文本文件的时候，ftell返回的可能不是文件字节计数，谨慎使用

# 指针函数与函数指针
函数指针：  
形如：int (*p)(形参，形参，，)为函数指针声明；

    p=函数名；//为函数指针赋值操作

p中存储的是函数地址

所以调用p应该为int a=（*p）（实参，实参，，，）

注意函数指针的括号不可以省略，否则就变为了指针函数

```c
#include<stdio.h>
#include<string.h>
//使用代码完成对某学生成绩的输出

int main(){
    float score[][4]={
        {},{20,30,40,50},{1,2,3,4},{5,5,6,8},{7,8,9,10}
    };
    float *search(float (*point)[4],int n);//指针函数声明
    
    float*(*q)(float (*point)[4],int n);//函数指针声明
    q=search;//函数指针初始化
    
    int n,m;
    printf("输入查询的学生号:");
    scanf("%d",&m);
    
    float *p=(*q)(score,m);//使用函数指针调用函数
    for(int i=0;i<4;i++){
        printf("%f\n",*(p+i));
    }
}
float *search(float(*point)[4],int n){
    float *t;
    t=*(point+n);//取得某行元素的首地址，而不是point+n，则需要俩次寻址
    return t;
}

```

# math 头文件
| 1 | [double pow(double x, double y)](https://www.runoob.com/cprogramming/c-function-pow.html)<br/><font style="color:rgb(51, 51, 51);">   </font><font style="color:rgb(51, 51, 51);">返回 x 的 y 次幂。</font> |
| --- | --- |
| <font style="color:rgb(51, 51, 51);">2</font> | [double sqrt(double x)](https://www.runoob.com/cprogramming/c-function-sqrt.html)<br/><font style="color:rgb(51, 51, 51);">   </font><font style="color:rgb(51, 51, 51);">返回 x 的平方根。</font> |
| <font style="color:rgb(51, 51, 51);">3</font> | [double ceil(double x)](https://www.runoob.com/cprogramming/c-function-ceil.html)<br/><font style="color:rgb(51, 51, 51);">   </font><font style="color:rgb(51, 51, 51);">返回大于或等于 x 的最小的整数值。</font> |
| <font style="color:rgb(51, 51, 51);">4</font> | [double fabs(double x)](https://www.runoob.com/cprogramming/c-function-fabs.html)<br/><font style="color:rgb(51, 51, 51);">   </font><font style="color:rgb(51, 51, 51);">返回 x 的绝对值。</font> |
| <font style="color:rgb(51, 51, 51);">5</font> | [double floor(double x)](https://www.runoob.com/cprogramming/c-function-floor.html)<br/><font style="color:rgb(51, 51, 51);">   </font><font style="color:rgb(51, 51, 51);">返回小于或等于 x 的最大的整数值。</font> |
| <font style="color:rgb(51, 51, 51);">6</font> | **<font style="color:rgb(77, 77, 77);">round()</font>**<font style="color:rgb(77, 77, 77);">： 四舍五入成整数</font><br/>```c round(2.3)=2, round(-2.5)=-3 ```  |
| <font style="color:rgb(51, 51, 51);">7</font> | <font style="color:rgb(198, 120, 221);">int </font><font style="color:rgb(97, 174, 238);">abs </font><font style="color:rgb(153, 153, 153);">(</font><font style="color:rgb(198, 120, 221);">int</font><font style="color:rgb(171, 178, 191);background-color:rgb(40, 44, 52);"> n </font><font style="color:rgb(153, 153, 153);">);</font><br/>求整数的绝对值 |


