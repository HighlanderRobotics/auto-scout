import Foundation
import Vision
import AppKit
let directory = CommandLine.arguments[1]
var rows:[[String:Any]]=[]
for name in try FileManager.default.contentsOfDirectory(atPath:directory).filter({$0.hasSuffix(".jpg")}).sorted() {
 guard let image=NSImage(contentsOfFile:directory+"/"+name)?.cgImage(forProposedRect:nil,context:nil,hints:nil) else { continue }
 let request=VNRecognizeTextRequest();request.recognitionLevel = .accurate;request.usesLanguageCorrection=false
 try VNImageRequestHandler(cgImage:image).perform([request])
 rows.append(["file":name,"reads":(request.results ?? []).compactMap { observation -> [String:Any]? in
  guard let candidate=observation.topCandidates(1).first else{return nil}
  return ["text":candidate.string,"confidence":candidate.confidence]
 }])
}
let data=try JSONSerialization.data(withJSONObject:rows,options:[.prettyPrinted,.sortedKeys]);FileHandle.standardOutput.write(data)
