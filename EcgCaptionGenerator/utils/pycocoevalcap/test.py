from eval import COCOEvalCap

COCOeval = COCOEvalCap()

gts = {'00015d9b2edf55e': ['<start> sinusritme <end>'],
 '00024b28a898873': ['<start> sinustachycardie rechter bundeltakblock <end>'],
 '0003939414ff3a4': ['<start> ventriculair gepaced ritme <end>'],
 '00055489cf6cf7b': ['<start> sinusbradycardie verder <end>'],
 '00063ac18b5556f': ['<start> sinusritme <end>'],
 '0006c99017dd4c4': ['<start> sinusritme <end>'],
 '000701dc6a3c338': ['<start> sinusritme met atrium extrasystole rechter bundeltakblock linker anterior fascikelblock bifasciculair block <end>'],
 '0007caa006144d6': ['<start> sinusbradycardie non-specifieke intraventriculaire geleidingsstoornis verder <end>'],
 '00086ef37bb0a6a': ['<start> sinusritme <end>'],
 '0009050c39cce63': ['<start> sinustachycardie non-specifieke st-afwijking <end>']}

res = {'00015d9b2edf55e': ['<start> sinusritme <end>'],
 '00024b28a898873': ['<start> sinustachycardie bundeltakblock <end>'],
 '0003939414ff3a4': ['<start> atriumfibrilleren gepaced <end>'],
 '00055489cf6cf7b': ['<start> sinusbradycardie <end>'],
 '00063ac18b5556f': ['<start> sinusritme <end>'],
 '0006c99017dd4c4': ['<start> sinusritme <end>'],
 '000701dc6a3c338': ['<start> sinusritme atrium linker rechter <end>'],
 '0007caa006144d6': ['<start> sinusbradycardie <end>'],
 '00086ef37bb0a6a': ['<start> sinusritme laag voltage <end>'],
 '0009050c39cce63': ['<start> sinustachycardie overweeg leasie acuut infarct de laterale <end>']}

COCOeval.evaluate(gts, res)