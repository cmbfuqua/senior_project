class fraction():
    def __init__(self):
        self.num = 0
        self.denom = 1
    def display(self):
        print(f'{self.num}/{self.denom}')
    def display_dec(self):
        print(self.num/self.denom)
    def User_input(self):
        self.num = int(input('Num: '))
        self.denom = int(input('Denom: '))

def main():
    frac = fraction()
    frac.display()
    frac.User_input()
    frac.display()
    frac.display_dec()

if __name__ == '__main__':
    main()
    