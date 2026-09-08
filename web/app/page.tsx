import { Dashboard } from "./Dashboard";
import type { UiBundle } from "@/lib/types";
import bundle from "../data/ui_bundle.json";

export default function Page() {
  return <Dashboard data={bundle as UiBundle} />;
}
