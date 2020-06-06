#6 함수 기초

#6_1
vege = "potato"
n = [4,5,2,7,6]
#6_2
print(len(vege)) #6
print(len(n))#5

#6_3 append 메서드의 예
alphabet = ["a", "b", "c", "d", "e"]
alphabet.append("f")
print(alphabet)   #['a', 'b', 'c', 'd', 'e', 'f']
#append 는 더해준다는 뜻이다 위의 예제처럼 하면 저 알파벳 리스트에 f라는 알파벳을 더해준것이 된다. 

#6_4 sorted
number = [1,5,3,4,2]
print(sorted(number))
print(number)
# [1, 2, 3, 4, 5]
# [1, 5, 3, 4, 2]
# sorted will not change the original number so if you would to print number after a sorted parameter it will still output a number before the sorted parameter.

#6_5 sort
number.sort()
print(number)#[1, 2, 3, 4, 5]
# sort will do the same thing but the difference is that it will now save the parameter number as the following.



#6_6 upper(), count()
city = "Tokyo"
print(city.upper())#TOKYO
#this will change the whole letters in capital letters and it only works whent it is in letters str

print(city.count("o"))#2
#this will count the "o" in the given letter

# 6_7
animal= "elephant"
print(animal)
# save the word elephant in the word animal in capital letters to animal_big

# count the "e"inside the letter animal

#6_8
animal_big = animal.upper()
print(animal_big)# ELEPHANT
print(animal.count("e"))# 2

#6_9 (format)
print("I was born in {} and spent the most of my life in {}".format("Busan", "Indonesia"))
#I was born in Busan and spent the most of my life in Indonesia

#6_10 Question
fruit = "Banana"
color = "Yellow"
#print something with the following message

#6_11 Answer
print("The color of {} is {}.".format(fruit, color))
#The color of my Banana is Yellow.

#6_12
alphabet = ["a", "b", "c", "d", "d"]
print(alphabet.index("a")) #0

print(alphabet.count("d")) #d
#the index number of alphabet is zero since the counting of the index number starts with zero.
#this means that the "d" is in the index number of 3

#6_13 
n = [3,6,8,6,3,2,4,6]
# print the index number of "2"
#print how many "6" are in n

#6_14 

print(n.index(2)) #5 , 2is in the index number of 5
print(n.count(6))   #3

#6_15 example of the "sort" method 
list = [1,10,2,20]
list.sort()
print(list)   #[1, 2, 10, 20]
#the number has been sorted in order

#6_16 reverse()
list = ["a", "b", "c", "d", "e"]
list.reverse()
print(list) #['e', 'd', 'c', 'b', 'a']
#the listed would be printed in the reverse order


#6_17_Q 
n = [53, 26,37,69,24,2]
#print the enumbers in  "n" in order
# print N in reverse order

#6_18 
n.sort()
print(n)#[2, 24, 26, 37, 53, 69]

n.reverse()
print(n)#[69, 53, 37, 26, 24, 2]

#6_19 
def sing():
    print("im singing")
sing()

#6_20_q 
# make a definition named introduce and to print "my name is mr. Hong" in it
def introduce():
    print("my name is Yoon Yung Son and I am so glad to meet you")
introduce()


#6_21 
def introduce():
    print("my name is Yoon Yung Son and I am so glad to meet you")
introduce()
#my name is Yoon Yung Son and I am so glad to meet you

#6_22  The factor element
def introduce(n):
    print(n + "am")
introduce("s")
# the answer would be sam, this factor element would input whatever is inside the n and connected to whatever element we have already defined.

#6_23 
#save cub_cal as time squared and print it


#6_24 
def cube_cal(n):
    print(n**2)
cube_cal(4)
#16

#6_25 
def introduce(first, second):
    print("MA Last name is " + first +", and my first name is "+second+ " you'd better remember the name")
introduce("Keum", "Do Hoon")


#6_26 Q
#print introduce



#6_27 
def introduce(n, age):
    print("My name is "+ n +" my age is " + str(age))
introduce("dohoon", 18)

#6_28 
def introduce(first = "kim", second ="chi"):
    print("ma first name is " + first + "and my last name is "+ second)
introduce("lol")


#6_29
def introduce(first = "kim", second ="chi"):
    print("ma first name is " + first + "and my last name is "+ second)
introduce(first)
#6_30 
def introduce(first = "kim", second):
    print("ma first name is " + first + "and my last name is "+ second)
introduce(first, second)


#6_31
# set the first value
def introduce(age, n):
    print(n + "is" + str(age) + "lol")

# this will lead to an error since there is no defined values for each of them (n and age)

#6_32 
def introduce(age, n=" my age"):
    print(n + " is " + str(age) + " lol")
introduce(18)  #my age is 18 lol

#6_33 the return
def introduce(first = "kim", second = "chi"):
    return "my first name is " + first + " and my last name is " + second

print(introduce("gal")) #my first name is galand my last name is chi


#6_34 
def introduce(first = "kim", second = "hong"):
    comment = "last name is "+ first +"and my first name is " +second
    return comment
print(introduce("gildong"))

#6_35 Q
#write a fraction that calculates the bmi and return bmi 

 
#6_36 answer
def bmi(height, weight):
    return weight / height**2
bmi(1.65, 65)
print(bmi)
#6_37 
#import time package
import time
#use time module to input the current time in now time
now_time = time.time()

#use print()
print("time", now_time)
#time 1591366864.445777


#6_38
#use from to import the time module "time"
from time import time
#since we have already imported the module we can skip in package name and use the module directly
now_time = time()

print(now_time)#1591367058.1061053

#6_39  Q
#use "from" to "import" "time"
from     import

#use "now_time" to import the current time
now_time =
print(now_time)

#6_40 A
# use "from" to "import" the "time" module
from time import time

#imput the current time in "now_time"

now_time = time()

#6_41 # class "member " and "method"
# we can save a value
mylist = [1, 10, 2, 20]

# we can sort the saved values
mylist.sort()

#we can change it to a method and print out the results
print(mylist)

#6_42
#define the class of "MyProduct"
Class MyProduct:

#define the creator
def __init__(self, name, price):
    #save the factors in member
    self.name = name
    self.price = price
    self.stock = 0
    self.sales = 0
#this is just a build up structure so you will have to print out the class in order to make something happen

#6_43
#bring "MyProduct" and make product1
product1 = MyProduct("cake", 500)


#6_44 Question
# Define the MyProduct class
Class MyProduct:
    #commit change 
    def __init__():
        #save the factor in the member


#We have to make product_1 by bringing MyProduct
product_1 = MyProduct("cake", 500, 20)

#print the stock of product_1 
print()

#6_45 Answer
Class MyProduct:
    def __init__(self, name, price, stock):
        self.name = name
        self.price = price
        self.stock = 0
        self.sales = 0        
product_1 = MyProduct("cake", 500, 20)
print(product_1.stock)




#6_46
class MyProduct:
    def __init__(self, name, price, stock):
        self.name=name
        self.price = price
        self.stock = 0
        self.sales = 0
    #buying method
    def buy_up(self, n):
        self.stock += n
    #selling method
    def sell(self, n):
        self.stock -=n
        self.sales += n*self.price
    #summary method
    def summary(self):
        message = "called summary()./n name: "+ self.name + /
        "/n price :" +str(self.price) + /
        "/n stocke :" +str(self.stock) + /
        "/n sales :" +str(self.sales) 
        print(message)




#6_47  Question 
class MyProduct:
    def __init__(self, name, price, stock):
        self.name=name
        self.price = price
        self.stock = 0
        self.sales = 0
    def summary(self):
        message = "called summary()./n name: "+ self.name + /
        "/n price :" +str(self.price) + /
        "/n stocke :" +str(self.stock) + /
        "/n sales :" +str(self.sales) 
        print(message)
    # to get the name
    def get_name():
    # discount by the number of factors
    def discount():
        #discount as much as 5000
#print the summary of product 2
product_2 = MyProduct("phone", 30000, 100)

#6_48 Answer
class MyProduct:
    def __init__(self, name, price, stock):
        self.name=name
        self.price = price
        self.stock = 0
        self.sales = 0
    def summary(self):
        message = "called summary()./n name: "+ self.name + /
        "/n price :" +str(self.price) + /
        "/n stocke :" +str(self.stock) + /
        "/n sales :" +str(self.sales) 
        print(message)
    # to get the name
    def get_name(self):
        return self name
    # discount by the number of factors
    def discount(self, n):
        self.price -= n

product_2 = MyProduct("phone", 30000, 100)
product_2.discount(5000)
product_2.summary

#discount as much as 5000
#print the summary of product 2

#6_49
class MyProductSalesTax(MyProduct):
    def __init__(self, name, price, stock, tax_rate):
        super() .__init__(name, price, stock)
        self.tax_rate = tax_rate
    def get_name(self):
        return self.name + "(소비세 포함)"
    def get_price_with_tax(self):
        return int(self.price * (1+self.tax_rate))
#6_50
class MyProductSalesTax("phone", 30000, 100, 0.1)
print(product_3.get_name())
print(product_3.get_price_with_tax())
product_3.summary

#6_51
class MyProduct:
    def __init__(self, name, price, stock):
        self.name=name
        self.price = price
        self.stock = 0
        self.sales = 0
    def summary(self):
        message = "called summary()./n name: "+ self.name + /
        "/n price :" +str(self.price) + /
        "/n stocke :" +str(self.stock) + /
        "/n sales :" +str(self.sales) 
        print(message)
    # to get the name
    def get_name(self):
        return self name
    # discount by the number of factors
    def discount(self, n):
        self.price -= n
class MyProductSalesTax(MyProduct):
    def __init__(self, name, price, stock, tax_rate):
        super() .__init__(name, price, stock)
        self.tax_rate = tax_rate
    def get_name(self):
        return self.name + "(소비세 포함)"
    def get_price_with_tax(self):
        return int(self.price * (1+self.tax_rate))
class MyProductSalesTax("phone", 30000, 100, 0.1)
print(product_3.get_name())
print(product_3.get_price_with_tax())
product_3.summary()

#6_52
def summary(self):
    message = "called summary(./n name: " +self.get_name() + /
        "/n price: " + str(self.get_price_with_tax()+0) + /
        "/n price: " + str(self.stock) + /
        "/n price: " + str(self.sales) 
    print(message)

#6_53 how to define words
pai = 3.141592
print("원주율은 %f" %pai)
print("원주율은 %.2f" %pai)
# 원주율은 3.141592
# 원주율은 3.14
#6_54 An bmi question
def bmi(height, weight):
    return weight / height**2
print("bmi는__입니다." %____)



#6_55 BMI answer

def bmi(height, weight):
    return weight / height**2
print("bmi는 %.4f 입니다." % bmi(1.65,65))
# bmi는 23.8751 입니다.

#6_56 Question
# print "check_character"

#print what to check inside the check charactier


#6_57 Answer
# print "check_character"
def check_character(object, character):
    return object.count(character)
#print what to check inside the check charactier
print(check_character([1,3,4,5,6,4,3,2,1,3,3,4,3], 3))
print(check_character("asdgaoirnoiafvnwoeo", "d"))
# 5
# 1

#6_58
#write down the equation from the "binary_search"
def binary_search(numbers, target_number):

   #Data for search
   numbers = [1,2,3,4,5,6,7,8,9,10,11,12,13]
   #value that is in search
   target_number = 11
   #activate binary search
   binary_search(numbers, target_number)


#6_59
def binary_search(numbers, target_number):
    #write down the estimated lowest number
    low = 0
    #highest number within the range
    high = len(numbers)
    #loop until found
    while low <= high:
        #find the middle number(index)datetime A combination of a date and a time. Attributes: ()
        middle = (low+high) //2
        #numbers(검색 대상)의 중앙값과 target_number(찾는 값)이 동일한 경우
        if numbers[middle] == target_number:
            #print
            print("{1}은 {0}번째에 있습니다.".format(middle, target_number))
            #ending
            break
        #if the middle number is less than the target number
        elif numbers[middle] < target_number:
            low = middle + 1
        #if the middle number is greater than the target number
        else:
            high = middle -1
numbers = [1,2,3,4,5,6,7,8,9,10,11,12,13]            
        















