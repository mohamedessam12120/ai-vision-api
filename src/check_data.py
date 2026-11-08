from torchvision import datasets

trainval = datasets.OxfordIIITPet(root="data", split="trainval", download=True)
test = datasets.OxfordIIITPet(root="data", split="test", download=True)

print("trainval images:", len(trainval))
print("test images:", len(test))
print("classes:", len(trainval.classes))
print("example classes:", trainval.classes[:5])
