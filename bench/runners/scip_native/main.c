#include "scip/pub_event.h"
#include "scip/pub_tree.h"
#include "scip/scip.h"
#include "scip/scipdefplugins.h"
#include "scip/scipgithash.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TRACE_EVENTS (SCIP_EVENTTYPE_PRESOLVEROUND | SCIP_EVENTTYPE_NODEFOCUSED | \
   SCIP_EVENTTYPE_NODEFEASIBLE | SCIP_EVENTTYPE_NODEINFEASIBLE | SCIP_EVENTTYPE_NODEBRANCHED | \
   SCIP_EVENTTYPE_DUALBOUNDIMPROVED | SCIP_EVENTTYPE_FIRSTLPSOLVED | SCIP_EVENTTYPE_LPSOLVED | \
   SCIP_EVENTTYPE_BESTSOLFOUND)

struct SCIP_EventhdlrData
{
   FILE* trace;
   SCIP_Real sampleinterval;
   SCIP_Real lastsample;
};

typedef struct
{
   const char* input;
   const char* stats;
   const char* trace;
   const char* solution;
   const char* summary;
   const char* log;
   SCIP_Real timelimit;
   SCIP_Real sampleinterval;
   int nparams;
   const char** params;
} OPTIONS;

static const char* eventName(SCIP_EVENTTYPE eventtype)
{
   if( eventtype == SCIP_EVENTTYPE_PRESOLVEROUND )
      return "presolve_round";
   if( eventtype == SCIP_EVENTTYPE_NODEFOCUSED )
      return "node_focused";
   if( eventtype == SCIP_EVENTTYPE_NODEFEASIBLE )
      return "node_feasible";
   if( eventtype == SCIP_EVENTTYPE_NODEINFEASIBLE )
      return "node_infeasible";
   if( eventtype == SCIP_EVENTTYPE_NODEBRANCHED )
      return "node_branched";
   if( eventtype == SCIP_EVENTTYPE_DUALBOUNDIMPROVED )
      return "dual_bound_improved";
   if( eventtype == SCIP_EVENTTYPE_FIRSTLPSOLVED )
      return "first_lp_solved";
   if( eventtype == SCIP_EVENTTYPE_LPSOLVED )
      return "lp_solved";
   if( eventtype == SCIP_EVENTTYPE_BESTSOLFOUND )
      return "best_solution_found";
   return "unknown";
}

static SCIP_Bool importantEvent(SCIP_EVENTTYPE eventtype)
{
   return eventtype == SCIP_EVENTTYPE_PRESOLVEROUND
      || eventtype == SCIP_EVENTTYPE_DUALBOUNDIMPROVED
      || eventtype == SCIP_EVENTTYPE_FIRSTLPSOLVED
      || eventtype == SCIP_EVENTTYPE_BESTSOLFOUND;
}

static void printJsonReal(SCIP* scip, FILE* file, SCIP_Real value)
{
   if( !isfinite(value) || SCIPisInfinity(scip, REALABS(value)) )
      fputs("null", file);
   else
      fprintf(file, "%.17g", value);
}

static SCIP_DECL_EVENTINIT(eventInitTrace)
{
   SCIP_CALL( SCIPcatchEvent(scip, TRACE_EVENTS, eventhdlr, NULL, NULL) );
   return SCIP_OKAY;
}

static SCIP_DECL_EVENTEXIT(eventExitTrace)
{
   SCIP_CALL( SCIPdropEvent(scip, TRACE_EVENTS, eventhdlr, NULL, -1) );
   return SCIP_OKAY;
}

static SCIP_DECL_EVENTEXEC(eventExecTrace)
{
   SCIP_EVENTHDLRDATA* data;
   SCIP_EVENTTYPE eventtype;
   SCIP_Real now;
   SCIP_NODE* node;
   int depth;

   (void)eventdata;
   data = SCIPeventhdlrGetData(eventhdlr);
   eventtype = SCIPeventGetType(event);
   now = SCIPgetSolvingTime(scip);

   if( !importantEvent(eventtype) && now - data->lastsample < data->sampleinterval )
      return SCIP_OKAY;
   data->lastsample = now;

   if( eventtype == SCIP_EVENTTYPE_PRESOLVEROUND )
   {
      fprintf(data->trace,
         "{\"event\":\"%s\",\"time_s\":%.9g,\"variables\":%d,\"constraints\":%d}\n",
         eventName(eventtype), now, SCIPgetNVars(scip), SCIPgetNConss(scip));
      fflush(data->trace);
      return SCIP_OKAY;
   }

   if( SCIPgetStage(scip) != SCIP_STAGE_SOLVING )
      return SCIP_OKAY;

   node = SCIPgetCurrentNode(scip);
   depth = node == NULL ? -1 : SCIPnodeGetDepth(node);
   fprintf(data->trace,
      "{\"event\":\"%s\",\"time_s\":%.9g,\"nodes\":%lld,\"depth\":%d,"
      "\"primal_bound\":",
      eventName(eventtype), now, (long long)SCIPgetNNodes(scip), depth);
   printJsonReal(scip, data->trace, SCIPgetPrimalbound(scip));
   fputs(",\"dual_bound\":", data->trace);
   printJsonReal(scip, data->trace, SCIPgetDualbound(scip));
   fputs(",\"gap\":", data->trace);
   printJsonReal(scip, data->trace, SCIPgetGap(scip));
   fprintf(data->trace,
      ",\"solutions\":%d,\"lp_solves\":%lld,\"lp_iterations\":%lld,"
      "\"strong_branch_calls\":%lld,\"strong_branch_lp_iterations\":%lld,"
      "\"separation_rounds\":%d,\"cuts_found\":%d,\"cuts_applied\":%d}\n",
      SCIPgetNSols(scip), (long long)SCIPgetNLPs(scip), (long long)SCIPgetNLPIterations(scip),
      (long long)SCIPgetNStrongbranchs(scip), (long long)SCIPgetNStrongbranchLPIterations(scip),
      SCIPgetNSepaRounds(scip), SCIPgetNCutsFound(scip), SCIPgetNCutsApplied(scip));
   fflush(data->trace);
   return SCIP_OKAY;
}

static void usage(const char* program)
{
   fprintf(stderr,
      "usage: %s --input FILE --stats FILE --trace FILE --solution FILE --summary FILE "
      "--log FILE --time-limit SEC [--sample-interval SEC] [--set NAME=VALUE ...]\n",
      program);
}

static int parseReal(const char* text, SCIP_Real* value)
{
   char* endptr;
   errno = 0;
   *value = strtod(text, &endptr);
   return errno == 0 && endptr != text && *endptr == '\0';
}

static int parseOptions(int argc, char** argv, OPTIONS* options)
{
   int i;

   memset(options, 0, sizeof(*options));
   options->timelimit = 30.0;
   options->sampleinterval = 0.25;
   options->params = (const char**)calloc((size_t)argc, sizeof(*options->params));
   if( options->params == NULL )
      return 0;

   for( i = 1; i < argc; ++i )
   {
      const char* arg = argv[i];
      const char* value;

      if( i + 1 >= argc )
         return 0;
      value = argv[++i];
      if( strcmp(arg, "--input") == 0 )
         options->input = value;
      else if( strcmp(arg, "--stats") == 0 )
         options->stats = value;
      else if( strcmp(arg, "--trace") == 0 )
         options->trace = value;
      else if( strcmp(arg, "--solution") == 0 )
         options->solution = value;
      else if( strcmp(arg, "--summary") == 0 )
         options->summary = value;
      else if( strcmp(arg, "--log") == 0 )
         options->log = value;
      else if( strcmp(arg, "--time-limit") == 0 )
      {
         if( !parseReal(value, &options->timelimit) )
            return 0;
      }
      else if( strcmp(arg, "--sample-interval") == 0 )
      {
         if( !parseReal(value, &options->sampleinterval) )
            return 0;
      }
      else if( strcmp(arg, "--set") == 0 )
         options->params[options->nparams++] = value;
      else
         return 0;
   }

   return options->input != NULL && options->stats != NULL && options->trace != NULL
      && options->solution != NULL && options->summary != NULL && options->log != NULL;
}

static SCIP_RETCODE writeStatistics(SCIP* scip, const char* path)
{
   FILE* file = fopen(path, "w");
   if( file == NULL )
      return SCIP_WRITEERROR;
   SCIP_CALL_FINALLY( SCIPprintStatisticsJson(scip, file), fclose(file) );
   return SCIP_OKAY;
}

static SCIP_RETCODE writeSolution(SCIP* scip, const char* path)
{
   FILE* file = fopen(path, "w");
   if( file == NULL )
      return SCIP_WRITEERROR;
   SCIP_CALL_FINALLY( SCIPprintBestSol(scip, file, FALSE), fclose(file) );
   return SCIP_OKAY;
}

static SCIP_RETCODE writeSummary(SCIP* scip, const char* path)
{
   FILE* file;
   SCIP_SOL* bestsol;

   file = fopen(path, "w");
   if( file == NULL )
      return SCIP_WRITEERROR;
   bestsol = SCIPgetBestSol(scip);
   fprintf(file,
      "{\"scip_version\":\"%d.%d.%d\",\"scip_git_hash\":\"%s\","
      "\"status\":\"%s\",\"objective\":",
      SCIPmajorVersion(), SCIPminorVersion(), SCIPtechVersion(), SCIPgetGitHash(),
      SCIPstatusName(SCIPgetStatus(scip)));
   if( bestsol == NULL )
      fputs("null", file);
   else
      printJsonReal(scip, file, SCIPgetSolOrigObj(scip, bestsol));
   fputs(",\"dual_bound\":", file);
   printJsonReal(scip, file, SCIPgetDualbound(scip));
   fputs(",\"gap\":", file);
   printJsonReal(scip, file, SCIPgetGap(scip));
   fprintf(file,
      ",\"nodes\":%lld,\"lp_solves\":%lld,\"lp_iterations\":%lld,"
      "\"strong_branch_calls\":%lld,\"strong_branch_lp_iterations\":%lld,"
      "\"cuts_found\":%d,\"cuts_applied\":%d,\"solutions\":%d,\"solving_time_s\":%.9g}\n",
      (long long)SCIPgetNNodes(scip), (long long)SCIPgetNLPs(scip),
      (long long)SCIPgetNLPIterations(scip), (long long)SCIPgetNStrongbranchs(scip),
      (long long)SCIPgetNStrongbranchLPIterations(scip), SCIPgetNCutsFound(scip),
      SCIPgetNCutsApplied(scip), SCIPgetNSols(scip), SCIPgetSolvingTime(scip));
   fclose(file);
   return SCIP_OKAY;
}

static SCIP_RETCODE run(const OPTIONS* options)
{
   SCIP* scip = NULL;
   SCIP_EVENTHDLR* eventhdlr = NULL;
   SCIP_EVENTHDLRDATA eventdata;
   int i;

   eventdata.trace = fopen(options->trace, "w");
   if( eventdata.trace == NULL )
      return SCIP_WRITEERROR;
   eventdata.sampleinterval = options->sampleinterval;
   eventdata.lastsample = -SCIP_REAL_MAX;

   SCIP_CALL( SCIPcreate(&scip) );
   SCIP_CALL( SCIPincludeDefaultPlugins(scip) );
   SCIPsetMessagehdlrLogfile(scip, options->log);
   SCIPsetMessagehdlrQuiet(scip, TRUE);
   SCIP_CALL( SCIPsetRealParam(scip, "limits/time", options->timelimit) );
   SCIP_CALL( SCIPsetIntParam(scip, "display/verblevel", 5) );
   SCIP_CALL( SCIPsetIntParam(scip, "display/freq", 1) );
   for( i = 0; i < options->nparams; ++i )
   {
      char* assignment = strdup(options->params[i]);
      char* equals;
      if( assignment == NULL )
         return SCIP_NOMEMORY;
      equals = strchr(assignment, '=');
      if( equals == NULL )
      {
         free(assignment);
         return SCIP_PARAMETERUNKNOWN;
      }
      *equals = '\0';
      SCIP_CALL( SCIPsetParam(scip, assignment, equals + 1) );
      free(assignment);
   }

   SCIP_CALL( SCIPincludeEventhdlrBasic(scip, &eventhdlr, "qopttrace",
      "q-opt SCIP white-box event trace", eventExecTrace, &eventdata) );
   SCIP_CALL( SCIPsetEventhdlrInit(scip, eventhdlr, eventInitTrace) );
   SCIP_CALL( SCIPsetEventhdlrExit(scip, eventhdlr, eventExitTrace) );

   SCIP_CALL( SCIPreadProb(scip, options->input, NULL) );
   SCIP_CALL( SCIPsolve(scip) );
   SCIP_CALL( writeStatistics(scip, options->stats) );
   SCIP_CALL( writeSolution(scip, options->solution) );
   SCIP_CALL( writeSummary(scip, options->summary) );

   fclose(eventdata.trace);
   SCIPsetMessagehdlrLogfile(scip, NULL);
   SCIP_CALL( SCIPfree(&scip) );
   BMScheckEmptyMemory();
   return SCIP_OKAY;
}

int main(int argc, char** argv)
{
   OPTIONS options;
   SCIP_RETCODE retcode;

   if( !parseOptions(argc, argv, &options) )
   {
      usage(argv[0]);
      free(options.params);
      return 2;
   }
   retcode = run(&options);
   free(options.params);
   if( retcode != SCIP_OKAY )
   {
      SCIPprintError(retcode);
      return (int)retcode;
   }
   return 0;
}
