
def calculator(n1,operator,n2):
    result = 0
    if operator == '*':
        result = n1*n2
    elif operator == '/':
        result = n1/n2
    elif operator == '+':
        result = n1+n2
    elif operator == '-':
        result = n1-n2
    return result    

def my_calculator():
    escape = False
    current_value = ""
    num1 = 0
    num2 = 0
    operator = ""
    while not escape:
        get_num1 = False
        if current_value != "":
            num1 = current_value
            get_num1 = True 
            print(f'\nFirst term is : {current_value}')
        
        while not get_num1:
            first_term = input("\nEnter first term : ") 
            if first_term == "x":
                escape = True
                break
            elif first_term == 'c':
                current_value = 0
                break
            elif first_term.isdigit():   
                num1 = int(first_term)
                get_num1 = True
        
        get_operator = False
        while get_num1 and not get_operator:
            operator = input('Enter operator : ')
            if operator == "x":
                escape = True
                break
            elif operator == 'c':
                current_value = 0
                break
            elif operator in ['+','-','/','*']:
                get_operator = True

        get_num2 = False        
        while get_operator and not get_num2:
            num2 = input("Enter second term : ")
            if num2 == "x":
                escape = True
                break
            elif num2 == 'c':
                current_value = 0
                break
            elif num2.isdigit():
                num2 = int(num2)
                get_num2 = True   

        if get_num2: 
            current_value = calculator(num1,operator,num2)
            print(f'{num1} {operator} {num2} = {current_value}')
    print('\nThe calculator has ended!\n')

def main():
    print('''\n\n-------- MY CALCULATOR------------ \n  ('c' to erase or 'x' to escape)''')
    my_calculator()

if __name__== "__main__":
    main()