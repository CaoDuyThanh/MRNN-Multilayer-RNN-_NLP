import csv
import collections

class DataHelper():
    def __init__(self,
                 fileName):
        self.FileName = fileName

        self.loadData()
        self.parseData()

    def loadData(self):
        print ('Read dataset from disk! File name = %s' % (self.FileName))
        self.AllChars = open(self.FileName, 'r').read()
        print ('Read data...Done! ')


    def parseData(self):
        charCount = collections.Counter(self.AllChars)
        self.NumChars = charCount.__len__()
        self.IdxToCharacter = [x[0] for x in charCount.iteritems()]
        self.CharacterToIdx = dict([(char, i) for i, char in enumerate(self.IdxToCharacter)])

        print ('Information about dataset')
        print ('     Number of characters = %d' % (self.AllChars.__len__()))
        print ('     List of characters = %s' % ((' ').join(self.IdxToCharacter)))

        # Convert all datato number
        self.AllCharsIdx = [self.CharacterToIdx[char] for char in self.AllChars]
        self.StartIdx = 0

    def NextBatch(self, length):
        if self.StartIdx + length > self.AllCharsIdx.__len__():
            subData = self.AllCharsIdx[self.StartIdx : self.AllCharsIdx.__len__()]
            subData.extend(self.AllCharsIdx[0 : length - (self.AllCharsIdx.__len__() - self.StartIdx)])
            self.StartIdx = (self.StartIdx + 1) % self.AllChars.__len__()
        else:
            subData = self.AllCharsIdx[self.StartIdx : self.StartIdx + length]
            self.StartIdx = (self.StartIdx + 1) % self.AllChars.__len__()
        output = subData[1:]
        output.append(self.AllCharsIdx[(self.StartIdx + length - 1) % self.AllChars.__len__()])

        return [subData, output]

        # if self.StartIdx + length > self.AllCharsIdx.__len__():
        #     subData = self.AllCharsIdx[self.StartIdx : self.AllCharsIdx.__len__()]
        #     subData.extend(self.AllCharsIdx[0 : length - (self.AllCharsIdx.__len__() - self.StartIdx)])
        #     self.StartIdx = (self.StartIdx + length) % self.AllCharsIdx.__len__()
        # else:
        #     subData = self.AllCharsIdx[self.StartIdx : self.StartIdx + length]
        #     self.StartIdx = (self.StartIdx + length) % self.AllCharsIdx.__len__()
        # output = self.AllCharsIdx[self.StartIdx]
        # return [subData, output]