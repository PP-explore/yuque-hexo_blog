---
title: Python
date: '2023-04-02 14:00:00'
updated: '2025-08-22 16:22:28'
---
# 一、python如何运行程序
python解释器（python标准实现形式）

1. python：也是一个名为**<font style="color:#DF2A3F;">解释器的软件包</font>**
    1. 解释器是一种让程序运行起来的程序
    2. 解释器是代码和计算机硬件之间的软件逻辑层
2. 解释器在程序执行的流程中起到了非常重要的作用。当程序被加载到内存中后，解释器会对程序进行解释执行，将程序中的指令翻译成机器指令，然后由CPU执行。

在该流程中，解释器的位置在编译器和CPU之间。编译器将源代码翻译成目标代码，然后将目标代码转换成可执行文件并存储在硬盘上。而解释器则在程序运行时将可执行文件中的指令解释成机器指令，然后交给CPU执行。

解释器的作用是实现程序的解释执行。相比编译执行，解释执行具有更好的灵活性和交互性。解释器可以在运行时进行错误检查和调试，还可以动态地修改程序的行为。因此，在一些需要动态性和交互性的应用中，解释器被广泛地应用。例如，解释器被广泛用于编程语言的交互式环境、脚本语言的执行和调试、以及一些特定领域的应用中。

<font style="color:rgb(52, 53, 65);">3.在程序执行流程中可执行文件和机器指令的区别</font>

       可执行文件和机器指令都是用来指示计算机执行任务的。但它们之间有一些区别。

       可执行文件是由源代码编译而来的，它是一种高级语言的二进制形式，通常具有跨平台的特性。可执行文件在操作系统中被视为一种数据文件，当用户运行可执行文件时，操作系统会将其加载到内存中，并将其解释成机器指令，最终交由CPU执行。在可执行文件中，不同的指令可以通过不同的操作码和操作数进行编码。

      机器指令是CPU能够直接执行的二进制指令，它是计算机的底层语言。它由操作码和操作数组成，每一条指令都直接控制计算机中的硬件资源，例如寄存器、内存、输入输出设备等。机器指令是计算机执行程序的最终形式，因为它是CPU能够直接理解和执行的指令。

       因此，在程序从创建到进入CPU执行的流程中，可执行文件是程序源代码编译后生成的中间产物，它需要被操作系统解释成机器指令才能被CPU执行。而机器指令是CPU能够直接执行的指令，是程序最终被执行的形式。解释器在该流程中的作用就是将可执行文件转换成机器指令，让CPU能够直接执行。

## 程序执行
+ 从程序员角度：一个python程序仅是一个包含python语言的文本文件。在解释器中运行这些代码
+ 从python内部角度：
    - 在代码开始处理之前，python还会执行一些程序。
    - 第一步，编译成所谓的“字节码”，这些字节码可以提高执行速度
        * 编译是一个简单的步骤，而且字节码是源代码底层的、与平台无关的表现形式。与机器指令不同，字节码不是直接由硬件执行的，而是由解释器或即时编译器在运行时将其转换为机器代码执行。
        * 字节码和机器的二进制代码都是计算机能够理解和执行的指令，它们的共同点在于都是以二进制的形式表示计算机指令。但是，它们之间也有一些区别。首先，<font style="background-color:#FBDE28;">字节码是一种中间代码，需要通过解释器或者即时编译器等虚拟机技术来解释和执行</font>，而机器的二进制代码则是直接由计算机处理器硬件执行的指令。其次，字节码是一种跨平台的代码格式，可以在多种计算机体系结构上运行，而机器的二进制代码则是特定于某种体系结构的，不同的计算机体系结构需要不同的机器指令。最后，由于字节码通常比机器的二进制代码更高级，所以在某些情况下，使用字节码可以减少程序的体积，同时也可以提高程序的安全性，因为字节码在执行之前需要经过解释器等中间层处理，可以对字节码进行各种安全检查和控制。
    - 第二步，将其转发到所谓的”python虚拟机“中
        * PVM实际上不是一个独立的程序，不需要安装。其实就是迭代运行字节码指令的一个大循环，一个接一个的完成操作，是python的运行引擎，是实际运行脚本的组件。技术上说，才是python解释器的最后一步。
        * PVM循环仍需解释字节码，并且字节码指令与CPu指令相比需要更多的工作



所以纯python代码的性能介于传统的编译语言和解释语言之间



## 执行模块的变体（python实现的替代方法）
1. CPython：原始的标准的python实现方式
2. Jython：目的是为了与java编程语言集成。jython包含java类，这些类编译python源代码、形成JAva字节码，并映射到java虚拟机中，可以通过python语句执行java语言代码
3. IronPython：目的是让python与windows平台上的。NET框架和Linux的Mono编写的应用相集成、



# 二、python对象类型
+  python全景：
    1. 程序由模块组成
    2. 模块包含语句
    3. 语句包含表达式
    4. 表达式建立并处理对象
+ python程序处理的每一个东西都是一个对象
+ python没有类型声明，运行的表达式的语法决定了创建和使用的对象的类型
+ <font style="background-color:#FBDE28;">一旦创建了一个对象，他就和操作集合绑定了</font>——如字符串值只可以进行字符串相关的操作
+ Python是动态类型（会自动跟踪你的类型）强类型语言（只能对一个对象进行该类型的 操作）
+ Python核心数据类型的优秀特性：支持任意的嵌套，并可以多个层次的嵌套



## 不可变性与可变性：
    1. <font style="background-color:#FBDE28;">数字、元组、字符串具有不可变性</font>——在被创建后就不可改变。但可以通过建立一个新的字符串并以相同变量名对其赋值
    2. 列表、字典可变

## 序列：
字符串：不可变对象，元素只能是字符 

列表：可变对象，元素类型不限 

元组：不可变对象，元素类型不限  

    1. python能反向索引：

S[-1]为从右开始的第一个索引项

    2. [ ]我们能在方括号内使用任意表达式，而不仅仅是数字常量
    3. 支持“分片”（Slice）的操作：

S[I : J],表示“取出在S中从偏移量为I，直到但不包含偏移量为J的内容”，结果返回一个新的对象

## ![](/images/0525686dbd5b8d0ed2ed93bc8a878be0.png)
## set 对象：
+ 不是序列，它是可变对象，但是<font style="color:#ED740C;">元素只能是不可变</font>类型。  
+  set 解析表达式： 

{c*4 for c in 'abcd'} 生成一个新 set 对象， 结果为： {'aaaa','bbbb','cccc','dddd'} （打印顺序不确定）  



## 1.字符串：（不可变序列）
+ 支持单引号双引号且表示相同
+ 支持三个引号中包括多行字符串常量：会默认在每行末尾添加\n换行符
+ 支持原始（raw）字符串常量，即去掉反斜线机制：

<font style="color:rgb(18, 18, 18);">在 Python 中，</font><font style="color:rgb(18, 18, 18);background-color:#FBDE28;">以字母 r 或者 R 作为前缀的字符串</font><font style="color:rgb(18, 18, 18);">，例如 r'...' 和 R'...'，被称为</font><font style="color:rgb(18, 18, 18);background-color:#FBDE28;">原始字符串</font><font style="color:rgb(18, 18, 18);">。与常规字符串不同，原始字符串中的反斜线（\）是一个普通字符，不具有转义功能。</font>

<font style="color:rgb(18, 18, 18);">原始字符串通常用于处理字符串中存在多个反斜线的情况，例如正则表达式和 Windows 目录路径。</font>

![](/images/40fe6b9e02bde35e17e08c359b39838f.png)

![](/images/b3c97e7562681ee0570129f78a540cd5.png)

![](/images/cc28fcfe4cabd66a466b01f5e6b3d293.png)

![](/images/b86a78b0c68ce8310dc201ad268ccacc.png)

+  str() ——将数字转成字符串，类似 print() 的效果 
+ repr()——产生的结果可以由解释器解释。 eval(repr(x)) 会返回 x 。  
+ ord(char) ：返回单个字符的 ASCII 值 
+ chr(ascii_int) ： 返回 ASCII 值对应的字符  
+ <font style="color:#1DC0C9;"> 字符串格式化表达式</font>是基于C语言的printf格式化表达式。其格式为："%d %s apples" % (3,'bad')  
+  .format()  ——字符串格式化表达式

![](/images/3073e57d003f40e81a05efa6915e19a3.png)





### 
### 
## 2.列表：没有固定类型的约束、没有固定大小
类型特定的操作：（都原地修改了原列表）

+ .**<font style="color:#DF2A3F;">list(iter_obj)</font>**—— 生成新列表  
+ .**<font style="color:#DF2A3F;">append(val)</font>**——在列表尾部插入一项
+ **<font style="color:#DF2A3F;">.pop(index)</font>**——移除索引位置的项
+ .**<font style="color:#DF2A3F;">pop()</font>**——删除末尾元素
+  **<font style="color:#DF2A3F;">.remove(val)</font>** ：通过值删除元素，若有多个值，则只删除第一个遇到的值  
+ **<font style="color:#DF2A3F;"> .insert(index,val)</font>** ：在指定位置插入元素，原地修改 
+ **<font style="color:#DF2A3F;"> index(val)</font>** ：<font style="color:#1DC0C9;">返回</font>指定元素的位置，若有多个值，则只返回第一个遇到的值所在位   
+ **<font style="color:#DF2A3F;">.sort()</font>**——默认按照升序排序
+ .**<font style="color:#DF2A3F;">reverse()</font>**——对列表翻转
+  .**<font style="color:#DF2A3F;">extend(iter_obj)</font>** 方法：在列表末端插入多个元素，原地修改  
+ .**<font style="color:#DF2A3F;">map(function,iterable,...)</font>**——为参数序列的每个元素调用function函数，返回包含每次function返回值的新列表



sorted(list1,key=None,reverse=False) ：

排序列表并返回新列表， <font style="color:#ED740C;">非原地修改</font> 

reversed(list1) ：

<font style="color:#ED740C;">返回迭代器</font>，该迭代器迭代结果就是<font style="color:#ED740C;">列表的逆序  </font>

![](/images/4dd6d413bfc4335faee7cc49794ef42e.png)





 索引和分片：

+  list1[index] ：索引，获取指定偏移处的对象， <font style="color:#1DC0C9;">并不会修改 list1 的值 </font>
+ list1[index1:index2] ：分片，返回一个新列表， 其元素值与旧列表对应片段处元素值相等， <font style="color:#1DC0C9;">并不会修改 list1 的值 </font>
+ 当索引位于赋值左侧时，则是索引赋值。这会改变列表指定项的内容
+ 当分片位于赋值左侧时，则是分片赋值。这会改变列表指定片段的内容
+ 被赋值的片断长度不一定要与赋值的片断长度相等  



边界检查：

python不允许引用超出列表末尾之外的索引值的元素，如S[1000]

嵌套：

M嵌套为一个三成三的矩阵，M【1】获取第二行的数据，

M【1】【2】获取第二行第三个数据

![](/images/b90b72000c8bde2e67cf352665c87f0b.png)

列表解析：（列表解析表达式）

源于集合的概念，通过对序列中的每一项运行一个表达式来创建一个新列表的方法。

+ <font style="color:#DF2A3F;">列表解析是编写在【】里的</font>（提醒你在创建一个列表的事实）

![](/images/fe67182f74a12704477c43d1503380d0.png)

实际应用中，列表解析可以更复杂

![](/images/092cbb307b733631602dbdface010fba.png)

![](/images/1c794cde91e43cad3e20455c11a44a24.png)



+ <font style="color:#DF2A3F;">括号中的解析语法</font>也可以用来创建产生所需结果的生成器：

例如，内置的sum函数，按一种顺序汇总各项

>>>G=(sum(row) for row in M)

>>>next(G)

6

>>>next(G)

15

用内置函数map也可完成：  
>>>map(sum,M)

[6,15,24]

+ 花括号中的解析语法也可创建集合和字典：  
![](/images/6ae5d47be67c2f4ac0de36d237d4d6e9.png)





## 3.字典：一种映射，而不是序列，存储一系列“键值对”
### 索引操作：和序列相同的用法
 <font style="color:#ED740C;">d[key] </font>——字典索引返回对应的值  

### 字典的生成：
<font style="color:#ED740C;">dict()</font>——从关键字参数生成字典： 

+ dict(a=3,b=4) 生成字典 {'a':3,'b':4}
+  可以通过 <font style="color:#ED740C;">zip() 函数</font>生成关键字参数： 

dict(zip(['a','b'],[3,4])) 生成字典 {'a':3,'b':4} 

 <font style="color:#ED740C;">.fromkeys()</font> 类方法生成字典： 

+ dict.fromkeys(['a','b']) 生成字典 {'a':None,'b':None} 
+ dict.fromkeys(['a','b'],3) 生成字典 {'a':3,'b':3}  

### 字典的迭代： 
<font style="color:#ED740C;">d.keys() </font>：返回一个dict_keys对象，它是一个可迭代对象，迭代时返回键序列 

<font style="color:#ED740C;">d.values()</font> ：返回一个dict_values对象，它是一个可迭代对象，迭代时返回值序列 

<font style="color:#ED740C;">d.items()</font> ：返回一个dict_items对象（，它是一个可迭代对象， 迭代时返回元组 (键，值) 的序列    



### 获取键的值：
通过<font style="color:#ED740C;"> </font>**<font style="color:#ED740C;">.get(key,default_value)</font>** 。返回键对应的值， 若键不存在则返回 default_value  

### 字典的操作： 
+ **<font style="color:#ED740C;">.update(d2)</font>** ：合并两个字典，原地修改 d1 字典 
+ **<font style="color:#ED740C;">.pop(key)</font>** ： 从字典中删除 key 并返回该元素的值 del
+ **<font style="color:#ED740C;">del d[key]</font>** ：从字典中删除 key 但是不返回该元素的值 
+ **<font style="color:#ED740C;">d[key]=value</font>** ：原地的添加/修改字典。当向不存在的键赋值时，相当于添加新元素  

### 重访嵌套：（python对象嵌套的应用）（核心类型的灵活性）
![](/images/7503af5b03425f00ec4f3ab161111011.png)

### 键的排序：for循环
+ 因为字典不是序列，不具有任何可靠的从左至右的顺序，意味着，它的键也许会以我们输入时不同的顺序出现
+ 常见的解决方法是：
    - 利用字典的.keys()方法收集一个键的列表
    - 使用列表的sort()方法进行排序
    - 用for循环逐个显示结果

```python
Ks=list(D.keys())
Ks.sort()
for key in Ks:
    print(key,'=>',D[key])

或者用sorted（）内置函数自动对字典排序
for key in sorted(D):
    print(key,'=>',D[key])
```



### ！！避免不存在的键：get（）、try语句、if/else表达式等等


## 4.元组（tuple）：类似不可改变的列表，有顺序的序列
+ 元组是序列，具有不可改变性，<font style="color:#ED740C;">编写在（）内</font>
+ 一旦创建不可改变
+ <font style="color:#ED740C;"> tuple(iter_obj) </font>函数从一个可迭代对象生成元组  
+ <font style="color:#ED740C;">.index(value)</font>——获取元组内某元素的索引号
+ .<font style="color:#ED740C;">count(value)</font>——某元素出现的次数
+ 支持混合的类型和嵌套
+ 支持索引和分片
+ 但是不支持索引赋值和分片赋值  



## 5.文件：文件对象是python代码对电脑上外部文件的主要接口
+ 没有特定的常量语法创建文件。需调用open函数以字符串的形式传递给它的一个外部的文件名和一个处理模式的字符串：

```python
例如创建一个文本输出文件，可以传递其文件名和‘w’处理模式字符串以便写入数据
f=open('test.txt','w')
f.write('hello\n')  返回值为输字符串的个数  6
f.wirte('yes')  返回值为输入字符串的个数   3
f.close()
f.readline() 读取下一行到一个字符串（包括行末的换行符）
f.readlines() ：按行读取接下来的整个文件到字符串列表，每个字符串一行
读取文件须先以‘r’处理模式（默认为r）open
在用read（）读取

如今读取一个文件的最佳方式就是根本不读它，
文件提供了一个迭代器（iterator）zaifor循环中或其他环境自动一行一行读取

文件使用方式标识

'r':默认值，表示从文件读取数据

'w':表示要向文件写入数据，并截断以前的内容

'a':表示要向文件写入数据，添加到当前内容尾部

'r+':表示对文件进行可读写操作(删除以前的所有数据)

'r+a'：表示对文件可进行读写操作(添加到当前文件尾部)

'b':表示要读写二进制数据。

'rb':表示用二进制方法读取
```

![](/images/327a732e42f7f1c3269d56aefded941f.png)

### os 模块中的文件描述符对象支持文件锁定之类的低级工具  
socket 、 pipe 、 FIFO 文件对象可以用于网络通信与进行同步
## 6.集合：唯一的不可变对象的无序集合
+ 用set（）创建
+ 支持一般的数学集合操作（如&，| ，-）
+ 

## 7.type类型和函数
type（某对象）——用来检测某对象的类型并且type自身是一个独立的type对象

 isinstance(x,typename) 用于测试 x 所指对象是否是 typename 类型  



##  8.None 对象：
+ 是一个特殊的Python对象，它总是 False ，一般用于占位。它有一块内存，是一个真正的对象。它 不代表未定义，事实上它有定义。  
+  None 是所有函数和方法的默认返回值  



## 9.数值类型
###  数字类型转换：
+  <font style="color:#ED740C;">hex(intx) 、 oct(intx) 、 bin(intx) 、 str(intx)</font> 将整数 intx 转换成十六/八/二/十进制表示的字符串 
+ <font style="color:#ED740C;">int(strx,base)</font> 将字符串 strx 根据指定的 base 进制转换成整数。 base 默认为10。
+  可以通过 <font style="color:#ED740C;">eval() </font>函数将字符串转为整数  
+ <font style="color:#ED740C;">float(strx)</font> 将字符串 strx 转换成浮点数 
+ <font style="color:#ED740C;">complex(num_real,num_imag) </font>创建一个复数，实部为数字 num_real ， 虚部为数字 num_imag  



 Python支持许多对数字处理的内置函数与内置模块：

 内置函数位于一个隐性的命名空间内，对应于 builtins 模块（python2.7叫做 __builtins__ 模块） 

+ math 模块：如 math.pi ， math.e ， math.sqrt …. 
+ 内置函数，如 pow() ， abs() ，…  



## 10.迭代器和生成器
 1.可迭代对象：在逻辑上它保存了一个序列，在迭代环境中依次返回序列中的一个元素值。  

 2.迭代协议： <font style="color:#ED740C;">.__next__() 方法</font>。 任何对象只要实现了迭代协议，则它就是一个迭代器对象 <font style="color:#ED740C;">迭代器对象调用 .__next__() 方法</font>，会得到下一个迭代结果 在一系列迭代之后到达迭代器尾部，若再次调用 .__next__() 方法，则会触发 StopIteration 异常 迭代器在Python中是用C语言的速度运行的，因此速度最快  

 3.Python3提供了一个内置的<font style="color:#ED740C;"> next() 函数</font>，它<font style="color:#ED740C;">自动调用迭代器的 .__next__() 方法</font>。即给定一个迭代器对 象 x ， next(x) 等同于 x.__next__()  

 4.内置的<font style="color:#EDCE02;"> iter()</font><font style="color:#DF2A3F;"> </font>函数用于从序列、字典、 set 以及其他可迭代<font style="color:#EDCE02;">对象中获取迭代器</font>。 对任何迭代器对象 iterator ，调用 <font style="color:#EDCE02;">iter(iterator) 返回它本身</font>  

# 类
## 类的定义：
```python
class class_name:
name1=val
def method(self):
pass
```

## class 语句内的顶层赋值语句会创建类的属性
+  class 语句创建的作用域会成为类属性的命名空间 如果是 class 内的 def 中的赋值语句，则并不会创建类的属性  

## 生成实例对象：instance_name=class_name()  
## 在类的 def 方法中，第一个参数（根据惯例称为 self ）会引用调用该函数的实例对象。
<font style="background-color:#FBDE28;">对 self 的属性赋值， 会创建或修改实例对象的属性</font>，而非类的属性。 

可以通过方法调用： instance_name.func() 

也可以通过类调用： class_name.func(instance_name)  

![](/images/d71a3a742cc5685b749b1aa2f57613aa.png)





## 类的继承
<font style="color:rgb(77, 77, 77);">继承是面向对象编程的一个核心概念。它允许我们定义一个继承另一个类的属性和方法的类。基类（父类）的特性被派生类（子类）继承。</font>

<font style="color:rgb(77, 77, 77);background-color:#FBDE28;">python中通过定义类名后面的括号，来继承括号里面的类</font>

```python
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        raise NotImplementedError("Subclasses must implement this method")

class Cat(Animal):
    def speak(self):
        return f"{self.name} says meow!"

# 创建Cat类的实例
my_cat = Cat("Whiskers")
print(my_cat.speak())  # 输出: Whiskers says meow!

```

### `super()` 的作用  
<font style="color:rgb(77, 77, 77);">在类的继承中，如果重定义某个方法，该方法会覆盖父类的同名方法，但有时，我们希望能同时实现父类的功能，这时，我们就需要调用父类的方法了。</font>  
<font style="color:rgb(77, 77, 77);">调用父类同名方法有两种方式：</font>  
<font style="color:rgb(77, 77, 77);">1、调用未绑定的父类方法</font>  
<font style="color:rgb(77, 77, 77);">2、使用super</font><font style="color:rgb(252, 85, 49);">函数</font><font style="color:rgb(77, 77, 77);">来调用</font>

**<font style="color:rgb(51, 51, 51);">super()</font>**<font style="color:rgb(51, 51, 51);"> </font><font style="color:rgb(51, 51, 51);">函数是用于调用父类(超类)的一个方法。</font>

**<font style="color:rgb(51, 51, 51);">super()</font>**<font style="color:rgb(51, 51, 51);"> </font><font style="color:rgb(51, 51, 51);">是用来解决多重继承问题的，直接用类名调用父类方法在使用单继承的时候没问题，但是如果使用多继承，会涉及到查找顺序（MRO）、重复调用（钻石继承）等种种问题。</font>

<font style="color:rgb(51, 51, 51);">MRO 就是类的方法解析顺序表, 其实也就是继承父类方法时的顺序表。</font>

### <font style="color:rgb(51, 51, 51);">语法</font>
<font style="color:rgb(51, 51, 51);">以下是 super() 方法的语法:</font>

`<font style="color:rgb(77, 77, 77);">super(type[, object-or-type])</font>`

### <font style="color:rgb(51, 51, 51);">参数</font>
+ <font style="color:rgb(51, 51, 51);">type -- 类。</font>
+ <font style="color:rgb(51, 51, 51);">object-or-type -- 类，一般是 self</font>

<font style="color:rgb(51, 51, 51);">Python3.x 和 Python2.x 的一个区别是: Python 3 可以使用直接使用 </font>**<font style="color:rgb(51, 51, 51);background-color:rgb(236, 234, 230);">super().xxx</font>**<font style="color:rgb(51, 51, 51);"> 代替 </font>**<font style="color:rgb(51, 51, 51);background-color:rgb(236, 234, 230);">super(Class, self).xxx</font>**<font style="color:rgb(51, 51, 51);"> :</font>

```python

class A(object):   # Python2.x 记得继承 object
    def add(self, x):
        y = x+1
        print(y)
class B(A):
    def add(self, x):
        super(B, self).add(x)
b = B()
b.add(2)  # 3
```

## <font style="color:rgb(25, 27, 31);">__init__</font>与 构造函数
<font style="color:rgb(25, 27, 31);">Python是一门面向对象的编程语言，面向对象是一种代码封装的技术，包含了各种功能，让代码能重复利用、高效节能。</font>

<font style="color:rgb(25, 27, 31);">通过class来定义类，类又包含了属性、方法等，</font><font style="color:rgb(25, 27, 31);background-color:#FBDE28;">属性是类里面的变量，方法是类里面的函数</font>

<font style="color:rgb(25, 27, 31);">__init__就是其中一种函数，叫做</font>[构造函数](https://www.zhihu.com/search?q=%E6%9E%84%E9%80%A0%E5%87%BD%E6%95%B0&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A2390824591%7D)

1. <font style="color:rgb(25, 27, 31);">每次创建类的实例对象时，__init__函数就会自动被调用，无论它里面有什么样的变量、计算，统统会自动调用。</font>
2. <font style="color:rgb(83, 88, 97);">带有两个下划线开头的函数是声明该属性为私有,不能在类地外部被使用或直接访问。	</font>
3. <font style="color:rgb(83, 88, 97);">init函数（方法）支持带参数的类的初始化 ，也可为声明该类的属性</font>
4. <font style="color:rgb(83, 88, 97);">init函数（方法）的第一个参数必须是 self（self为习惯用法，也可以用别的名字），后续参数则可 以自由指定，和定义函数没有任何区别。</font>

## <font style="color:rgb(34, 34, 38);">__call__()方法运用</font>
`<font style="color:rgb(199, 37, 78);background-color:rgb(249, 242, 244);">__call__()</font>`<font style="color:rgb(77, 77, 77);">是一种magic method，在类中实现这一方法可以使该类的实例（对象）像</font><font style="color:rgb(252, 85, 49);">函数</font><font style="color:rgb(77, 77, 77);">一样被调用。默认情况下该方法在类中是没有被实现的。使用callable()方法可以判断某对象是否可以被调用。</font>

<font style="color:rgb(77, 77, 77);">__call__()方法的作用其实是把一个类的实例化对象变成了可调用对象，也就是说把一个类的实例化对象变成了可调用对象，只要类里实现了__call__()方法就行。如当类里没有实现__call__()时，此时的对象p 只是个类的实例，不是一个可调用的对象，当调用它时会报错：‘Person’ object is not callable.</font>

![](/images/67e910bfa6f4f2e55d312c791745e23e.png)![](/images/fb08180f03ee52647c829ef88f69f1d4.png)

<font style="color:rgb(77, 77, 77);">现在把Person类的实例p变成可调用的对象</font>

![](/images/812d28163b05153312aebd8e077726fe.png)

![](/images/14d69624dab6de51959e2ebdd71375be.png)

**<font style="color:rgb(77, 77, 77);">单看 p(‘Tim’) 你无法确定 p 是一个函数还是一个类实例，所以，在Python中，函数也是对象，对象和函数的区别并不显著。</font>**

**<font style="color:rgb(77, 77, 77);"></font>**

# <font style="color:rgb(34, 34, 38);">Python类型注解(Typing Hint) </font>
### <font style="color:rgb(79, 79, 79);">什么是 Python 类型注解？</font>
使用静态类型的编程语言，例如 C/C++、Java，必须预先声明变量、函数参数和返回值类型。编译器在编译和运行之前会强制检查代码的类型定义是否符合要求。运行时不能随意改变类型，当然需要的话，C++指针变量可改变指向对象的类型，但必须显式地用cast()函数来实现。

而 Python 解释器使用动态类型，函数的变量、参数和返回值可以是任何类型。此外，在程序运行时，允许不加申明更改变量类型。Python 解释器根据上下文来动态推断变量类型。

```plain
def add(x,y)
    return x+y
print(add(10+20))
print(add(3.14+5.10))
print(add(10+20.33))
print(add("hello","world"))

```

从上面例子可以看出，动态类型的优点：使编程变得容易，代码可读性更佳。这也是Zen of Python(Python之禅)所倡导的：简单比复杂好，复杂比错综复杂好。但也有代价，因为灵活，容易出现因前后理解不一致而造成的错误, 如经常遇到的1个典型问题：传入SQL的数据与数据库期望的不一致而导致SQL操作失败。

###  Python3 类型注解的基本语法
Python3.5 引入了类型注解（typing hint）, 可以同时利用静态和动态类型二者优点。 语法上有些类似于 typescript 的类型注解，但python 的类型注解使用更加方便，强烈建议在项目开发中应用此功能, 可以帮助规避很多代码中的变量使用错误。

<font style="color:rgb(77, 77, 77);">给</font>[函数参数](https://so.csdn.net/so/search?q=%E5%87%BD%E6%95%B0%E5%8F%82%E6%95%B0&spm=1001.2101.3001.7020)<font style="color:rgb(77, 77, 77);">、返回值添加类型注解的语法为：</font>

```plain
(parameter: type)    # 函数参数类型注解
-> type              # 返回值类型注解

```

<font style="color:rgb(77, 77, 77);">例如，下面演示如何对函数的参数和返回值使用类型注解：</font>

```plain
def say_hi(name: str) -> str:
    return f'Hi {name}'


greeting = say_hi('John')
print((greeting)

```



# 异常
1.Python中，异常会根据错误自动地被触发，也能由代码主动触发和截获 2.捕捉异常的代码：  

![](/images/c05516005a7ca48112efda012b372c2a.png)

# [装饰器 - Python教程 - 廖雪峰的官方网站](https://liaoxuefeng.com/books/python/functional/decorator/index.html)
## 注册代码为方法


# 模块与库
## 文件操作
```plain
#解压

#打开文件：
open(filepath,mode）
#读取每一行返回的是列表
f.readlines()
#字符串切割
split("匹配的字符")
#删除一行的开头和结尾的空白字符（包括空格和换行符）
line.strip()
#shutil模块
#复制或移动到目的文件夹下面
shutil.copy('demo.txt','新的文件夹')
shutil.move('file1.txt','新的文件夹')
#复制或移动到目的文件夹下面之后，重命名
shutil.copy('demo.txt','新的文件夹/new1.txt')
shutil.move('file2.txt','新的文件夹/new2.txt')
```

文件的相关操作用用os模块![](/images/0c7b263b6607aba37d50a3dfdf818ae5.png)

```python
路径：
#获取绝对路径
os.path.abspath(path)
#路径拼接  ***
os.path.join(path1[,path2[,……]])
#把路径分割为文件所在目录路径(dirname)和文件名(basename)
os.path.split(path)
#获取文件的所在的目录
os.path.dirname(file_path)
#获取当前文件的所在目录
os.path.dirname(os.path.abspath(__file__))
#判断路劲是否存在
bool os.path.exists(path)
#判断是否为绝对路径
os.path.isabs(path)
#判断路径是否为目录
os.path.isdir(path)
#判断是否为文件
os.path.isfile(path)


```

## 正则
## Flask
```plain
#Flask 会在指定的主机和端口上持续运行并监听 HTTP 请求
self.app.run(host=self.host, port=self.port, debug=True, threaded=True)

```

## hashlib   哈希库
hashlib 是一个提供了一些流行的hash(摘要)算法的Python标准库．

其中所包括的算法有<font style="background-color:#FCE75A;"> md5, sha1, sha224, sha256, sha384, sha512, blake2b(), blake2s()</font>

什么是摘要算法呢？摘要算法又称哈希算法、散列算法。它通过一个函数，把任意长度的数据转换为一个长度固定的数据串（通常用16进制的字符串表示）



### <font style="color:rgb(180, 180, 180) !important;">1、方法</font>
```python
--update(data)：更新hash对象的字节数据，data是需要加密的文本内容，需要"转为bytes类型"，如果”abc123”.encode() 将字符串”abc123”转为bytes类型。

--digest()：返回加密后的bytes值内容，类型：bytes。

--hexdigest()：返回加密后的哈希值文本内容。

--copy()：复制hash 对象信息。

示例：
import hashlib

m = hashlib.md5("123456".encode())
print("打印md5密文-bytes类型：",m.digest())
print("打印md5密文-哈希值：",m.hexdigest())

.encode()：将字符串 key 转换为字节格式（bytes），因为哈希函数需要处理字节数据而不是字符串。
哈希函数返回的结果本质上是一个二进制数据块（256 位），通常以 "16 进制字符串"形式呈现
hexdigest()：将哈希结果以16进制字符串形式返回（64字符长的字符串）
int(..., 16):将hexdigest()得到的16进制字符串转为整数。这一步将哈希值转换为一个大整数，可以用于后续的计算。

```

## curl


## requests  库
+ **HTTP方法：**
+ `GET`：从服务器获取数据。
+ `POST`：向服务器发送数据。
+ `PUT`：更新服务器上的数据。
+ `DELETE`：删除服务器上的数据。
+ **URL组成部分：**
+ 协议：如 `http://` 或 `https://`
+ 域名：如 `www.example.com`
+ 路径：如 `/api/v1/resource`
+ 查询参数：如 `?key=value`
+ **状态码：**
+ `200`：请求成功
+ `404`：资源未找到
+ `500`：服务器内部错误





## socket
Socket 又称 "套接字"，应用程序通常通过 "套接字" 向网络发出请求或者应答网络请求，使主机间或者一台计算机上的进程间可以通讯。python中提供了两个基本的 Socket 模块：服务端 Socket 和客户端 Socket，当创建了一个服务端 Socket 后，这个 Socket 就会在本机的一个端口上等待连接，当客户端 Socket 访问这个端口，两者完成连接后就能够进行交互了。

```python
#Socket模块 服务器端常用API
sock = socket.socket()        #创建一个服务器套接字
sock.bind()                   #绑定端口
sock.listen()                 #监听连接
sock.accept()                 #接受新连接
sock.close()                  #关闭服务器套接字

#Socket模块  客户端客户端常用API
sock = socket.socket()        # 创建一个套接字
sock.connect()                # 连接远程的服务器
sock.recv()                   # 读消息
sock.send()                   # 写消息
sock.sendall()                # 完全写消息
sock.close()                  # 关闭连接

#注意: sendall()方法是对send()方法的封装，send()可能只发送部分内容，返回值为实际发送字节
#sendall()是对send()方法的封装，如果一次send没有发完，那就多发几次，直到发完或者出现错误返回
#所以一般使用sendall()方法比较好

#socket.recv(int)这个方法是传入长度值来读取指定长度字节的消息，它是阻塞调用的
#它是尽可能地将接收缓存中的内容取出就返回，并非是将等到期望的字节数全满足才返回
#实际上循环读取才是正确的
def receive(sock,n):
    rs = []
    while n > 0:   #循环读取字节
        r = sock.recv(n)
        if not r:
            return rs
        rs.append(r)
        n -= len(r)
    return ''.join(rs)
```

## struct库    <font style="color:rgb(145, 152, 161);">转换成字节流</font>
<font style="color:rgb(77, 77, 77);">完成Python数值和C语言结构体的Python字符串形式间的转换。</font>

<font style="color:rgb(77, 77, 77);">用于处理存储在文件中或从网络连接中存储的二进制数据，以及其他数据源。</font>

<font style="color:rgb(77, 77, 77);">意思就是主要功能是</font>

+ <font style="color:rgb(77, 77, 77);">将python的</font>**<font style="color:#262626;">数值类型</font>**<font style="color:rgb(77, 77, 77);">转化为字节序列</font>
+ <font style="color:rgb(77, 77, 77);">能够指定格式（就像是C的结构体一样的格式）进行字节转换</font>

### <font style="color:rgb(77, 77, 77);">1. 模块函数和Struct类</font>
<font style="color:rgb(77, 77, 77);">提供一个Struct类，还有许多模块级的函数</font>**<font style="color:rgb(77, 77, 77);">用于处理结构化的值</font>**<font style="color:rgb(77, 77, 77);">。</font>

<font style="color:rgb(77, 77, 77);">这里有个</font>**<font style="color:rgb(77, 77, 77);">格式符(Format specifiers)</font>**<font style="color:rgb(77, 77, 77);">的概念，是指从字符串格式转换为已编译的表示形式，类似于</font>[<font style="color:rgb(77, 77, 77);">正则表达式</font>](https://so.csdn.net/so/search?q=%E6%AD%A3%E5%88%99%E8%A1%A8%E8%BE%BE%E5%BC%8F&spm=1001.2101.3001.7020)<font style="color:rgb(77, 77, 77);">的处理方式。</font>

<font style="color:rgb(77, 77, 77);">通常实例化Struct类，调用类方法来完成转换，比直接调用模块函数有效的多。下面的例子都是使用Struct类。</font>

:::info
 当使用 `struct` 模块处理结构化数据时，实例化 `Struct` 类并使用其方法进行打包和解包，通常比直接调用模块级的函数（如 `struct.pack()` 和 `struct.unpack()`）更有效和更方便  。 当您实例化一个 `Struct` 类的对象后，格式字符串只需解析一次。每次调用实例的方法（如 `pack()` 和 `unpack()`）时，不需要重新解析格式字符串，从而提高性能。  

:::

### **<font style="color:rgb(77, 77, 77);">2. Packing（打包）和Unpacking（解包）</font>**
<font style="color:rgb(77, 77, 77);">Struct支持将数据packing(打包)成字符串，并能从字符串中逆向unpacking(解压)出数据。</font>

```python
import struct

import binascii

values = (1, 'ab'.encode('utf-8'), 2.7) #结构化的数据

s = struct.Struct('I 2s f')  #struct指定数据格式
packed_data = s.pack(*values) #struct打包成字节

print('原始值:', values)
print('格式符:', s.format)
print('占用字节:', s.size)
print('打包结果:', binascii.hexlify(packed_data))
# output
原始值: (1, b'ab', 2.7)
格式符: b'I 2s f'
占用字节: 12
打包结果: b'0100000061620000cdcc2c40'


s = struct.Struct('I 2s f')    
unpacked_data = s.unpack(packed_data)   #解压数据
print('解包结果:', unpacked_data)

# output
解包结果: (1, b'ab', 2.700000047683716)
```

<font style="color:rgb(77, 77, 77);">格式符对照表如下:</font>

![](/images/fdb012a460feafe1c04ed3401faa093b.png)

## json序列化库
<font style="color:#262626;">Python 内置的json序列化库，将内存中的对象序列化成json字符串，也可将字符串反序列化对象。</font>![](/images/5cceb1b0b9ab70f519614606d9eb5bb9.png)

```python
serializedData = json.dumps({"hello":"world"})  #序列化，把python类型的数据转换成json字符串
origin = json.loads(serializedData)             #反序列化，load的用法是把json格式文件，转换成python类型的数据。
```

**<font style="color:rgb(77, 77, 77);">JSON 对象可以写为如下形式：</font>**

```python
[{
    "name": "小明",
    "height": "170",
    "age": "18"
}, {
     "name": "小红",
    "height": "165",
    "age": "20"
}]

```

**<font style="color:rgb(77, 77, 77);">JSON 可以由以上两种形式自由组合而成，可以无限次嵌套，结构清晰，是数据交换的极佳方式。</font>**

### <font style="color:rgb(79, 79, 79);">loads，load的用法</font>
**<font style="color:rgb(77, 77, 77);">例如，有一段 JSON 形式的字符串，它是 str 类型，我们用 json.loads转换成python的数据结构，变成列表或字典，这样我们就能进行操作了。</font>**

```python
import json

data = '''
[{
    "name": "小明",
    "height": "170",
    "age": "18"
}, {
     "name": "小红",
    "height": "165",
    "age": "20"
}]
'''

# 打印data类型
print(type(data))
# json类型的数据转化为python类型的数据
new_data = json.loads(data)
# 打印data类型
print(type(new_data))

```

![](/images/bf1d98596f86d338a60e0257aa8be618.png)

## <font style="color:#262626;">asyncore    </font><font style="color:rgb(77, 77, 77);">异步通讯模块</font>
**<font style="color:rgb(77, 77, 77);">Python的asyncore模块提供了以异步的方式写入套接字服务的客户端和服务器的基础结构，下面一些常用的API和类</font>**

```python
asyncore.loop()函数：用于循环监听网络事件,它负责检测一个字典，字典中保存dispatcher实例
asyncore.dispatcher类：
    简介：一个底层套接字对象的封装，其中的writeable()和readable()在检测到一个socket
         可以写入或者数据到达的时候被调用，返回一个bool值，决定是否调用handle_send或者handle_write方法
    常用方法：
    	create_socket():创建一个socket连接
    	connect(address) : 连接一个socket_server
    	send(data):   发送数据
    	recv(buffer_size) ：  接收数据到内存缓冲
    	listen() :      serversocket开始监听
    	bind(address): server socket绑定某个地址或端口
    	accept():  等待客户端的连接
    	close()   关闭socket
     一般需要复写的方法：
        handle_read():当socket有可读数据时执行该方法，判定就是根据readable()返回True还是False
        handle_write():当socket有可写数据时调用这个方法 ,判定就是根据writeable()返回True还是False
        handle_connect():当客户端有连接的时候执行该方法进行处理
        handle_close():当连接关闭的时候执行该方法
        handle_accept():当作为server socket监听的时候，有客户端连接就使用该方法进行处理
asyncore.dispatcher_with_send类：一个dispatcher的子类，添加了简单的缓冲能力，对于简单客户端可以使用
```

## <font style="color:rgb(77, 77, 77);">BytesIO 和StringIO</font>
### <font style="color:#262626;">内存中的IO</font>
之前说的磁盘上的文件，就是将数据持久化到磁盘的一块区域，供后面重复使用。其优点就是持久化稳定不丢失，但是缺点也很明显，就是每次要使用都要从磁盘读入，相对内存而言很缓慢。

如果只是<font style="background-color:#FBDE28;">短时间的重复利用，并不希望长期持久化</font>，而且<font style="background-color:#FBDE28;">对速度的要求比较高</font>，这时候就可以<font style="background-color:#FBDE28;">考虑缓存</font>。说到缓存，很多朋友就想到redis，熟悉python的朋友还会想到装饰器和闭包函数。

不过python已经原生为我们准备好了<font style="color:#262626;background-color:#FBDE28;">类文件对象（file-like object）</font>，这种对象<font style="background-color:#FBDE28;">在内存中创建，可以像文件一样被操作。</font>

下面我们就来学习下两种类文件对象，StringIO和BytesIO。

### <font style="color:#262626;">标志位</font>
<font style="color:#262626;">内存中的对象有一个标志位的概念，往里面写入，标志位后移到下一个空白处。</font>

<font style="color:#262626;">而读数据的时候是从标志位开始读，所以</font>**<font style="color:#262626;">想要读取前面的数据需要手动将标志位进行移动。</font>**

```python
#常用的方法StringIO
s=StringIO()  
s.write('this\nis\na\ngreat\nworld!')
s.getvalue() 
s.read()     
s.seek(0)   
s.readline()  :以\n为分界读取单行
s.readlines() :以\n为分界读取所有行并存储在一个列表中
s.close()
#常用的方法ByteIO：
f = BytesIO()   :   实例化一个读写字节的对象
f.getvalue()    :   该方法用于获取写入后的字节串
f.write()       :   写入字节串
f.seek(0)  
f.read()        :   读入指定数目的字节串
f.tell()        :   用于获取当前文件的指针位置
f.seek(offset,whence):  用于移动文件读写指针到指定位置，第一个参数是偏移量，第二参数为0表示文件开头，
                        为2表示文件末尾，为1表示当前位置
f.close()
```

```python
利用requests库从网络下载一张图片
In [7]: response=requests.get('https://img.zcool.cn/community/0170cb554b9200000001bf723782e6.jpg@1280w_1l_2o_100sh.jpg')                                                     

In [8]: type(response.content)                                                                                                                                               
Out[8]: bytes
直接保存到BytesIO中
In [12]: img=BytesIO(response.content)

In [13]: from PIL import Image                                                                                                                                               
In [14]: pic=Image.open(img)                                                                                                                                                 
In [15]: pic.format                                                                                                                                                          
Out[15]: 'JPEG'


```

<font style="color:#262626;">一些不必永久存储在磁盘上的临时文件就可以直接放内存中使用了，不过和文件一样，使用完记得及时关闭回收内存空间。</font>

## <font style="color:#262626;">os.fork() 、os.kill(）、signal    子进程创建</font>
:::info
<font style="color:#262626;">Windows内核不支持这两个模块，所以服务端文件必须在类Unix内核的系统中运行</font>

:::

```python
#fork()函数用于生成一个子进程，这个函数会在父子进程同时返回，
#父进程中返回一个大于0的整数值，即为子进程的进程号
#子进程会返回0，如果返回值小于0说明系统资源不足，无法创建
pid = os.fork()
if pid > 0:
    #parent process
if pid == 0:
    #child process
if pid < 0:
    #出现错误

#使用fork创建子进程后，子进程会复制父进程的数据信息，而后程序就分两个进程继续运行后面的程序，
#这也是fork（分叉）名字的含义了。
#os.kill()函数，可以向指定进程发送某个信号
os.kill(pid,signal.SIGKILL)    #向某个进程发送SIGKILL信号，强制杀死进程，无法捕获，只能暴力退出
os.kill(pid,signal.SIGTERM)    #向某个进程发送SIGTERM信号，终止信号，可以自定义信号处理函数
os.kill(pid,signal.SIGINT)     #向某个进程发送SIGKINT信号，中断信号，可以捕获，自定义信号处理函数

#os.waitpid(pid,options)可以指定具体的pid来收割子进程，也可以通过pid = -1来收割任意子进程
os.waitpid(pid,options)
#options如果是0，就表示阻塞等待子进程结束才会返回，如果是WNOHANFG就表示非阻塞，有就返回进程pid,没有返回0
#这个函数有可能抛出子进程不存在异常，要进行异常处理

#常用信号
1.SIGINT：信号一般指代键盘的 ctrl+c 触发的 Keyboard Interrupt
2.SIGTERM：此信号默认行为也是退出进程，但是允许用户自定义信号处理函数。
3.SIGKILL：此信号的处理函数无法覆盖，进程收到此信号会立即暴力退出。
4.SIGCHLD：子进程退出时，父进程会收到此信号。当子进程退出后，父进程必须通过 waitpid 来收割子进程，
          否则子进程将成为僵尸进程，直到父进程也退出了，其资源才会彻底释放。
```

<font style="color:#262626;">signal包的核心就是使用signal.signal()函数来预设（register）信号处理函数</font>

```python
signal.signal(signalnum,handler) #其中signalnum为某个信号，handler为该信号的处理函数

我们在信号基础里提到，进程可以无视信号，可以采取默认操作，还可以自定义操作。
当handler为signal.SIG_IGN时，信号被无视(ignore)。
当handler为singal.SIG_DFL，进程采取默认操作(default)。
当handler为一个函数名时，进程采取函数中定义的操作。

signal.alarm(time) ：定时发出SIGALRM信号

```

## <font style="color:#262626;">errono包</font>
<font style="color:#262626;">该包定义了许多操作系统调用错误码，主要是帮助程序进行异常处理</font>

```python
errno.EPERM :Operation not permitted    # 操作不允许
errno.ENOENT:No such file or directory  # 文件没找到
errno.ESRCH:No such process             # 进程未找到
errno.EINTR:Interrupted system call     # 调用被打断
errno.EIO:I/O error                     # I/O 错误
errno.ENXIO:No such device or address   # 设备未找到
errno.E2BIG:Arg list too long           # 调用参数太多
errno.ENOEXEC:Exec format error         # exec 调用二进制文件格式错误
errno.EBADF:Bad file number             # 文件描述符错误
errno.ECHILD:No child processes         # 子进程不存在
errno.EAGAIN:Try again                  # I/O 操作被打断，告知 I/O 操作重试
```

## <font style="color:#262626;">kazoo包</font>
<font style="color:#262626;">kazoo是一个Python库，使得Python能够很好的使用zookeeper；</font>

<font style="color:#262626;">zookeeper是一个分布式的数据库，可以实现诸如分布式应用配置管理，统一命名服务，状态同步服务，集群管理功能。</font>

<font style="color:#262626;">在本次项目中我们主要使用监听通知机制完成服务发现的功能</font>

```python
#kazoo常用API
from kazoo.client import KazooClient
zk = KazooClient(hosts = "127.0.0.1:2181")   #默认情况下KazooClient会连接本地的2181端口，即zookeeper服务
zk.start()        #不断尝试连接
zk.stop()         #显式中断连接

#zookeeper节点的增删查改API
ensure_path():递归创建节点路径，不能添加数据
create():     创建节点，并同时可以添加数据，前提是父节点必须存在，不能递归创建
exists() :    检查节点是否存在
get():        获取节点数据以及节点状态的详细信息
get_children() ： 获取指定节点的所有子节点
set():        更新节点的信息
delete():     删除指定节点
    
#监听器  kazoo可以在节点上添加监听，使得节点或者子节点发生变化时进行出发
1.方式一：zookeeper原生支持
def test_watch_data(event):
    print("this is a watcher for node data")
zk.get_children("/demo",watch = test_watch_children)
2.使用Python的装饰器
@zk.ChildrenWatch("/demo")     #当子节点发生变化时出发
def watch_china_children(children):
    print("this is watch_child_children")
@zk.DataWatch("/demo")         #当节点数据发生变化出发
def watch_china_node(data,state):
    print("china node")
    
```

