# Canonical final archive

The experiment completed on 2026-09-15 at 11:32 +0800. Algorithm, parameters,
predictions and metrics are unchanged from implementation commit12ed4ff and
result commit16513c9. The previous result commit is retained on
exp/v10-ndp-prototype as an immutable publication record.

That archive hashed one Windows CRLF Markdown report before Git normalized it
to LF according to .gitattributes. Its report checksum therefore failed on
Linux. This final branch starts from the identical implementation commit,
copies the exact canonical Git blobs of every result, and computes the
manifest from those bytes. It does not rewrite the prior commit or its files.

Implementation branch: exp/v10-ndp-prototype.
Final validated archive branch: exp/v10-ndp-prototype-final.
The report's branch and publication references describe the original run;
this note identifies the final delivery branch. No experiment was rerun.
