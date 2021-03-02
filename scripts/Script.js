importClass(Packages.ij.IJ)
importClass(Packages.ij.gui.PointRoi)


imp = IJ.openImage("/home/ryan/Documents/GitHub/autoballs/scripts/temp.tif");
IJ.run(imp, "Convert to Mask", "");
imp.setRoi(new PointRoi(512,512,"small yellow hybrid"));
imp.show();
//IJ.run(imp, "Sholl Analysis (From Image)...", "startradius=0.0 stepsize=0.0 endradius=512.0 hemishellchoice=[None. Use full shells] previewshells=false nspans=5.0 nspansintchoice=Mean primarybrancheschoice=[Use no. specified below:] primarybranches=1.0 polynomialchoice=[Use degree specified below:] polynomialdegree=2.0 normalizationmethoddescription=[Automatically choose] normalizerdescription=Default plotoutputdescription=[Linear plot] tableoutputdescription=[Detailed table] annotationsdescription=[ROIs (points and 2D shells)] lutchoice=mpl-viridis.lut luttable=net.imglib2.display.ColorTable8@1d99d123 save=true savedir=/home/ryan/Documents/GitHub/temp analysisaction=[Analyze image]");
IJ.run(imp, "Sholl Analysis...", "starting=10 ending=512 radius_step=0 #_samples=5 integration=Mean enclosing=1 #_primary=0 infer fit linear polynomial=[Best fitting degree] most normalizer=Area create overlay save directory=/home/ryan/Documents/GitHub/autoballs/scripts/temp");
IJ.run("Close All", "");