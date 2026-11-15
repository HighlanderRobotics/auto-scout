// Local Apple Vision OCR. Input: directory of robot crops. Output: JSON evidence.
import Foundation
import Vision
import AppKit

let folder = URL(fileURLWithPath: CommandLine.arguments[1])
var output: [[String: Any]] = []
for file in try FileManager.default.contentsOfDirectory(atPath: folder.path).sorted() where file.hasSuffix(".jpg") {
    guard let image = NSImage(contentsOf: folder.appendingPathComponent(file))?.cgImage(forProposedRect: nil, context: nil, hints: nil) else { continue }
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = false
    request.minimumTextHeight = 0.015
    try VNImageRequestHandler(cgImage: image).perform([request])
    for observation in request.results ?? [] {
        // Only the best candidate: alternatives from one crop are not independent votes.
        guard let candidate = observation.topCandidates(1).first else { continue }
        let words = candidate.string.components(separatedBy: CharacterSet.decimalDigits.inverted).filter { !$0.isEmpty }
        for word in words {
            output.append(["file": file, "team": word, "confidence": candidate.confidence])
        }
    }
}
let data = try JSONSerialization.data(withJSONObject: output, options: [.prettyPrinted, .sortedKeys])
FileHandle.standardOutput.write(data)
